"""Spectrum readers for NMRPipe and processed Bruker data."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from nmrglue.fileio import bruker, pipe, sparky
from numpy.typing import NDArray

from peakfit.engine.domain.config import DQFCOSYWindow
from peakfit.engine.domain.param_id import PSEUDO_AXIS
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, get_dimension_label
from peakfit.engine.domain.window import ProcessingWindow, WindowKind

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


T = TypeVar("T", float, NDArray[Any])

P1_MIN = 175.0
P1_MAX = 185.0

# NMRPipe nucleus label mapping from header codes
NUCLEUS_LABELS: dict[str, str] = {
    "1": "1H",
    "2": "2H",
    "13": "13C",
    "15": "15N",
    "19": "19F",
    "31": "31P",
}
BRUKER_BINARY_NAMES = frozenset({"1r", "2rr", "3rrr", "4rrrr"})
_BRUKER_GM = 2
_BRUKER_EM = 1
_BRUKER_SINE = 3
_BRUKER_QSINE = 4
_BRUKER_TPPI = 3
_BRUKER_COMPLEX_MODES = {4, 5, 6}  # States, States-TPPI, Echo-Antiecho.
_BRUKER_COMPLEX_TRANSFORMS = {5, 6}
_MIN_SINE_SHIFT = 2.0
_QUADRATURE_DIMENSIONS = 2
_MIN_TIME_POINTS = 2


def _positive_int(value: Any) -> int | None:
    """Parse a positive integer from header value."""
    try:
        parsed = round(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def read_spectral_parameters(dic: dict[str, Any], data: NDArray[Any]) -> list[Any]:
    """Read spectral parameters from an NMRPipe dictionary."""
    spec_params: list[SpectralParameters] = []

    for i in range(data.ndim):
        size = data.shape[i]
        fdf = f"FDF{int(dic['FDDIMORDER'][data.ndim - 1 - i])}"
        is_direct = i == data.ndim - 1
        ft = dic.get(f"{fdf}FTFLAG", 0.0) == 1.0

        if i == 0:
            dim_label = PSEUDO_AXIS
            nucleus = None
            sw = obs = car = aq_time = 1.0
            p180 = False
        else:
            spectral_index = i - 1
            dim_label = get_dimension_label(spectral_index)
            nucleus_code = str(int(dic.get(f"{fdf}OBS", 0) % 100))
            label_key = f"{fdf}LABEL"
            if dic.get(label_key):
                nucleus = str(dic[label_key]).strip()
            else:
                nucleus = NUCLEUS_LABELS.get(nucleus_code)

            if ft:
                sw = dic.get(f"{fdf}SW", 1.0)
                orig = dic.get(f"{fdf}ORIG", 0.0)
                obs = dic.get(f"{fdf}OBS", 1.0)
                car = orig + sw / 2.0 - sw / size
                aq_time = dic.get(f"{fdf}APOD", 0.0) / max(sw, 1e-6)
                p180 = P1_MIN <= abs(dic.get(f"{fdf}P1", 0.0)) <= P1_MAX
            else:
                sw = obs = car = aq_time = 1.0
                p180 = False

        processing_window = None
        processing_error = None
        if i > 0 and ft:
            try:
                processing_window = _pipe_processing_window(dic, fdf, size)
                aq_time = processing_window.points * processing_window.dwell
            except ValueError as error:
                processing_error = str(error)

        spec_params.append(
            SpectralParameters(
                size=size,
                sw=sw,
                obs=obs,
                car=car,
                aq_time=aq_time,
                apocode=dic.get(f"{fdf}APODCODE", 0.0),
                apodq1=dic.get(f"{fdf}APODQ1", 0.0),
                apodq2=dic.get(f"{fdf}APODQ2", 0.0),
                apodq3=dic.get(f"{fdf}APODQ3", 0.0),
                p180=p180,
                direct=is_direct,
                ft=ft,
                label=dim_label,
                nucleus=nucleus,
                processing_window=processing_window,
                processing_window_error=processing_error,
                td_size=_positive_int(dic.get(f"{fdf}TDSIZE")),
                ft_size=_positive_int(dic.get(f"{fdf}FTSIZE")),
            )
        )

    return spec_params


def _pipe_processing_window(dic: dict[str, Any], prefix: str, size: int) -> ProcessingWindow:
    """Recover sampling before a standard frequency-domain EXT -sw extraction."""
    sw = _header_number(dic, f"{prefix}SW")
    points = _header_number(dic, f"{prefix}APOD")
    x1 = _float_value(dic.get(f"{prefix}X1"))
    xn = _float_value(dic.get(f"{prefix}XN"))
    if x1 or xn:
        ft_size = _positive_int(dic.get(f"{prefix}FTSIZE"))
        if ft_size is None or xn < x1 or not np.isclose(xn - x1 + 1, size):
            raise ValueError("extracted NMRPipe data requires FTSIZE and matching X1/XN")
        ratio = ft_size / size
        sw *= ratio
        points *= ratio
    if sw <= 0 or points < _MIN_TIME_POINTS or not np.isclose(points, round(points)):
        raise ValueError("cannot recover integral acquisition points from NMRPipe APOD/SW")
    code = _header_number(dic, f"{prefix}APODCODE")
    kinds: dict[float, WindowKind] = {0: "none", 1: "sine", 2: "em", 3: "gm", 7: "gmb"}
    if code not in kinds:
        raise ValueError(f"unsupported NMRPipe APODCODE={code:g}")
    q = [_header_number(dic, f"{prefix}APODQ{i}") for i in (1, 2, 3)]
    if code == 1 and (not 0 <= q[0] <= q[1] <= 1 or q[2] < 0):
        raise ValueError("unsupported sine window offsets or power")
    # NMRPipe stores C1 as the applied multiplier minus one.
    first = _header_number(dic, f"{prefix}C1") + 1
    dwell = 1 / sw
    duration = (round(points) - 1) * dwell if code in {1, 3} else round(points) * dwell
    return ProcessingWindow(
        kind=kinds[code],
        points=round(points),
        dwell=dwell,
        duration=duration,
        q1=q[0],
        q2=q[1],
        q3=q[2],
        first_point=first,
    )


def _bruker_processing_window(
    acquisition: dict[str, Any], processing: dict[str, Any], *, direct: bool
) -> ProcessingWindow:
    """Infer a window on acquired samples; predicted samples are deliberately excluded."""
    if _float_value(processing.get("TDoff")) != 0:
        raise ValueError("time-domain shifts (TDoff) are not supported")
    points, dwell, acquired = _bruker_time_sampling(acquisition, processing, direct=direct)
    code = _header_number(processing, "WDW")
    kinds: dict[float, WindowKind] = {0: "none", 1: "em", 2: "gmb", 3: "sine", 4: "sine"}
    if code not in kinds:
        raise ValueError(f"unsupported Bruker WDW={code:g}")
    q1 = q2 = q3 = 0.0
    if code in {1, 2}:
        q1 = _header_number(processing, "LB")
    if code == _BRUKER_GM:
        q2 = _header_number(processing, "GB")
        if q1 >= 0 or not 0 < q2 <= 1:
            raise ValueError("Bruker Gaussian multiplication requires LB < 0 and 0 < GB <= 1")
    if code in {3, 4}:
        shift = _header_number(processing, "SSB")
        if shift < 0:
            raise ValueError("SSB must be nonnegative")
        q1 = 1 / shift if shift >= _MIN_SINE_SHIFT else 0.0
        q2 = 1.0
        q3 = 1.0 if code == _BRUKER_SINE else 2.0
    return ProcessingWindow(
        kind=kinds[code],
        points=points,
        dwell=dwell,
        duration=acquired * dwell,
        q1=q1,
        q2=q2,
        q3=q3,
        first_point=_header_number(processing, "FCOR") if "FCOR" in processing else 0.5,
    )


def _is_bruker_processed_directory(path: Path) -> bool:
    return (path / "procs").is_file() and any(
        (path / filename).is_file() for filename in BRUKER_BINARY_NAMES
    )


def resolve_spectrum_path(path: Path) -> Path:
    """Return a readable NMRPipe file or canonical Bruker processing directory."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.name in BRUKER_BINARY_NAMES:
            return candidate.parent
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Spectrum path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Spectrum path is not a file or directory: {candidate}")
    if _is_bruker_processed_directory(candidate):
        return candidate

    pdata = candidate if candidate.name == "pdata" else candidate / "pdata"
    if not pdata.is_dir():
        raise ValueError(
            "A Bruker spectrum directory must contain procs and 1r/2rr/3rrr/4rrrr, "
            "or contain a pdata directory."
        )

    processed = sorted(
        (
            item
            for item in pdata.iterdir()
            if item.is_dir() and _is_bruker_processed_directory(item)
        ),
        key=lambda item: (
            not item.name.isdigit(),
            int(item.name) if item.name.isdigit() else item.name,
        ),
    )
    preferred = pdata / "1"
    if preferred in processed:
        return preferred
    if len(processed) == 1:
        return processed[0]
    if not processed:
        raise ValueError(f"No processed Bruker spectrum was found below {pdata}.")
    choices = ", ".join(item.name for item in processed)
    raise ValueError(
        f"Multiple processed Bruker spectra were found below {pdata}: {choices}. "
        "Select a specific pdata/<procno> directory."
    )


def _pseudo_parameters(size: int) -> SpectralParameters:
    return SpectralParameters(
        size=size,
        sw=1.0,
        obs=1.0,
        car=1.0,
        aq_time=1.0,
        apocode=0.0,
        apodq1=0.0,
        apodq2=0.0,
        apodq3=0.0,
        p180=False,
        direct=False,
        ft=False,
        label=PSEUDO_AXIS,
        nucleus=None,
    )


def _bruker_axis_key(prefix: str, ndim: int, axis: int) -> str:
    bruker_dimension = ndim - axis
    if bruker_dimension == 1:
        return f"{prefix}s"
    return f"{prefix}{bruker_dimension}s"


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bruker_aq_time(
    acquisition: dict[str, Any], axis: dict[str, Any], sw: float, *, direct: bool
) -> float:
    """Derive acquisition duration without using a cropped frequency-domain width."""
    acquisition_sw = _bruker_acquisition_width(acquisition)
    if direct:
        acquisition_time = _float_value(acquisition.get("AQ"))
        if acquisition_time > 0.0:
            return acquisition_time

    td_size = _positive_int(acquisition.get("TD"))
    if np.isfinite(acquisition_sw) and acquisition_sw > 0.0:
        sw = acquisition_sw
    if td_size is None or sw <= 0.0:
        return 1.0
    points = td_size / 2.0 if bool(axis.get("complex", True)) else float(td_size)
    return points / sw


def _bruker_acquisition_width(acquisition: dict[str, Any]) -> float:
    """Read the acquisition sweep width in Hz, before spectral extraction."""
    sw = _float_value(acquisition.get("SW_h"))
    if np.isfinite(sw) and sw > 0.0:
        return sw
    return _float_value(acquisition.get("SW")) * _float_value(acquisition.get("SFO1"))


def _header_number(parameters: dict[str, Any], name: str) -> float:
    value = _float_value(parameters.get(name), float("nan"))
    if not np.isfinite(value):
        raise ValueError(f"missing or invalid {name}")
    return value


def _bruker_time_sampling(
    acquisition: dict[str, Any], processing: dict[str, Any], *, direct: bool
) -> tuple[int, float, int]:
    """Return retained samples, dwell seconds, and acquired samples before zero filling."""
    # guess_udic(strip_fake=True) deliberately changes SW for extracted spectra.
    # Always derive timing from the acquisition dictionary instead.
    sw = _bruker_acquisition_width(acquisition)
    if not np.isfinite(sw) or sw <= 0:
        raise ValueError("missing or invalid acquisition SW_h or SW and SFO1")
    td = _header_number(acquisition, "TD")
    if td <= 0 or not td.is_integer():
        raise ValueError("TD must be a positive integer")

    if direct:
        # Simultaneous QSIM/DQD acquisition stores interleaved real/imaginary points.
        mode = _header_number(acquisition, "AQ_mod")
        if mode not in {1, 3}:
            raise ValueError("DQF-COSY requires complex QSIM or DQD direct acquisition")
        paired = True
    else:
        mode = _float_value(acquisition.get("FnMODE"))
        if mode == 0 and "MC2" in processing:
            mode = _header_number(processing, "MC2") + 1
        if mode == 0 and processing.get("FT_mod") in _BRUKER_COMPLEX_TRANSFORMS:
            mode = 4  # The status transform establishes complex quadrature pairing.
        if mode not in _BRUKER_COMPLEX_MODES and mode != _BRUKER_TPPI:
            raise ValueError("unsupported or missing indirect FnMODE/MC2 for phase-sensitive data")
        paired = mode in _BRUKER_COMPLEX_MODES

    raw_points = int(td)
    tdeff = _float_value(processing.get("TDeff"))
    if not np.isfinite(tdeff) or tdeff < 0 or not tdeff.is_integer():
        raise ValueError("TDeff must be a nonnegative integer")
    retained = min(raw_points, int(tdeff)) if tdeff > 0 else raw_points
    transform_size = _positive_int(processing.get("FTSIZE") or processing.get("SI"))
    if transform_size is not None:
        retained = min(retained, 2 * transform_size)
    divisor = 2 if paired else 1
    if raw_points % divisor or retained % divisor:
        raise ValueError("TD and TDeff must contain complete real/imaginary pairs")
    samples = retained // divisor
    if samples < _MIN_TIME_POINTS:
        raise ValueError("fewer than two retained time-domain samples")
    # TPPI acquires real samples at half the complex dwell interval.
    dwell = 1 / sw if paired else 1 / (2 * sw)
    return samples, dwell, raw_points // divisor


def _bruker_dqfcosy_window(
    acquisition: dict[str, Any], processing: dict[str, Any], *, direct: bool
) -> DQFCOSYWindow:
    """Translate supported TopSpin status parameters to the sampled fitting window."""
    if _float_value(processing.get("ME_mod")) != 0:
        raise ValueError("linear prediction (ME_mod) is outside the DQF-COSY model")
    if _float_value(processing.get("TDoff")) != 0:
        raise ValueError("time-domain shifts (TDoff) are outside the DQF-COSY model")
    points, dwell, acquired_points = _bruker_time_sampling(acquisition, processing, direct=direct)
    code = _header_number(processing, "WDW")
    if code not in {0, _BRUKER_EM, _BRUKER_SINE, _BRUKER_QSINE}:
        raise ValueError(f"unsupported Bruker window WDW={code:g}")
    power = 0
    offset = 0.0
    end = 1.0
    lb = 0.0
    if code == _BRUKER_EM:
        lb = _header_number(processing, "LB")
    elif code in {_BRUKER_SINE, _BRUKER_QSINE}:
        shift = _header_number(processing, "SSB")
        if shift < 0:
            raise ValueError("SSB must be nonnegative")
        offset = 1 / shift if shift >= _MIN_SINE_SHIFT else 0.0
        power = 1 if code == _BRUKER_SINE else 2
        # TopSpin defines sin(phi + (pi-phi)*t/AQ). Convert the angle at
        # the final retained sample to our explicit endpoint convention.
        end = offset + (1 - offset) * (points - 1) / acquired_points
    return DQFCOSYWindow(
        acquisition_time=points * dwell,
        points=points,
        sine_power=power,
        sine_offset=offset,
        sine_end=end,
        lb_hz=lb,
    )


def _read_bruker_spectral_parameters(
    dic: dict[str, Any], data: NDArray[Any], udic: dict[Any, Any], first_is_pseudo: bool
) -> list[SpectralParameters]:
    params = [_pseudo_parameters(data.shape[0])]
    original_ndim = data.ndim if first_is_pseudo else data.ndim - 1
    first_spectral_axis = 1 if first_is_pseudo else 0

    for axis_index in range(first_spectral_axis, original_ndim):
        universal_axis = udic[axis_index]
        processing = dic.get(_bruker_axis_key("proc", original_ndim, axis_index), {})
        acquisition = dic.get(_bruker_axis_key("acqu", original_ndim, axis_index), {})
        spectral_index = axis_index - first_spectral_axis
        direct = axis_index == original_ndim - 1
        sw = _float_value(universal_axis.get("sw"), 1.0)
        obs = _float_value(universal_axis.get("obs"), 1.0)
        car = _float_value(universal_axis.get("car"))
        window_code = round(_float_value(processing.get("WDW")))
        nucleus = str(universal_axis.get("label") or "").strip("<> ") or None
        window = None
        window_error = None
        try:
            window = _bruker_dqfcosy_window(acquisition, processing, direct=direct)
        except ValueError as error:
            # Generic spectrum loading also supports windows outside the DQF model.
            # Defer the actionable error until that model is actually requested.
            window_error = str(error)

        processing_window = None
        processing_error = None
        try:
            processing_window = _bruker_processing_window(acquisition, processing, direct=direct)
        except ValueError as error:
            processing_error = str(error)

        params.append(
            SpectralParameters(
                size=int(universal_axis.get("size", data.shape[axis_index])),
                sw=sw,
                obs=obs,
                car=car,
                aq_time=_bruker_aq_time(acquisition, universal_axis, sw, direct=direct),
                # Bruker and NMRPipe window codes are not equivalent. Use the exact
                # no-window model only for WDW=0 and the robust pvoigt fallback otherwise.
                apocode=0.0 if window_code == 0 else -1.0,
                apodq1=0.0,
                apodq2=0.0,
                apodq3=0.0,
                p180=P1_MIN <= abs(_float_value(processing.get("PHC1"))) <= P1_MAX,
                direct=direct,
                ft=True,
                label=get_dimension_label(spectral_index),
                nucleus=nucleus,
                td_size=_positive_int(acquisition.get("TD")),
                ft_size=_positive_int(processing.get("FTSIZE") or processing.get("SI")),
                processing_window=processing_window,
                processing_window_error=processing_error,
                dqfcosy_window=window,
                dqfcosy_window_error=window_error,
            )
        )

    return params


def _read_bruker(
    path: Path, binary_name: str | None = None
) -> tuple[dict[str, Any], NDArray[Any], list[SpectralParameters], bool]:
    dic, original_data = (
        bruker.read_pdata(str(path))
        if binary_name is None
        else bruker.read_pdata(str(path), bin_files=[binary_name])
    )
    original_data = np.asarray(original_data)
    udic = bruker.guess_udic(dic, original_data, strip_fake=True)

    first_processing = dic.get(_bruker_axis_key("proc", original_data.ndim, 0), {})
    first_is_pseudo = (
        original_data.ndim > 1 and round(_float_value(first_processing.get("FT_mod"), 1.0)) == 0
    )
    data = original_data if first_is_pseudo else np.expand_dims(original_data, axis=0)
    params = _read_bruker_spectral_parameters(dic, data, udic, first_is_pseudo)
    dic["_peakfit_format"] = "bruker"
    return dic, data, params, not first_is_pseudo


def read_spectra(
    path_spectra: Path,
    path_z_values: Path | None = None,
    exclude_list: Sequence[int] | None = None,
    *,
    use_imaginary: bool = False,
    imaginary_paths: Mapping[Literal["ri", "ir", "ii"], Path] | None = None,
) -> Spectra:
    """Read an NMRPipe or processed Bruker spectrum and optional plane values."""
    resolved_path = resolve_spectrum_path(path_spectra)
    if resolved_path.is_dir():
        dic, data, params, pseudo_dim_added = _read_bruker(resolved_path)
    elif resolved_path.suffix.lower() in {".ucsf", ".sparky"}:
        dic, original = sparky.read(str(resolved_path))
        udic = sparky.guess_udic(dic, original)
        data = original[None, ...]
        params = [_pseudo_parameters(1)]
        for axis in range(original.ndim):
            meta = udic[axis]
            params.append(
                SpectralParameters(
                    size=meta["size"],
                    sw=meta["sw"],
                    obs=meta["obs"],
                    car=meta["car"],
                    aq_time=0,
                    apocode=-1,
                    apodq1=0,
                    apodq2=0,
                    apodq3=0,
                    p180=False,
                    direct=axis == original.ndim - 1,
                    ft=True,
                    label=get_dimension_label(axis),
                    nucleus=meta["label"],
                )
            )
        dic["_peakfit_format"] = "ucsf"
        pseudo_dim_added = True
    else:
        dic, data = pipe.read(resolved_path)
        data = data.astype(np.float32)
        udic = pipe.guess_udic(dic, data)
        first_is_freq = bool(udic.get(0, {}).get("freq", True))
        if first_is_freq:
            data = np.expand_dims(data, axis=0)
        params = read_spectral_parameters(dic, data)
        pseudo_dim_added = first_is_freq

    if path_z_values is not None:
        z_values = np.genfromtxt(path_z_values, dtype=None, encoding="utf-8")
    else:
        z_values = np.arange(data.shape[0], dtype=int)

    spectra = Spectra(
        dic=dic,
        data=data,
        z_values=z_values,
        params=params,
        pseudo_dim_added=pseudo_dim_added,
    )
    if imaginary_paths and not use_imaginary:
        raise ValueError("Imaginary component paths require --use-imaginary.")
    if use_imaginary:
        if path_z_values is not None or exclude_list:
            raise ValueError("Quadrature fitting cannot use --z-values or --exclude.")
        return _read_quadrature_components(spectra, resolved_path, imaginary_paths)
    spectra.exclude_planes(exclude_list)

    return spectra


def _read_quadrature_components(
    real: Spectra, resolved_path: Path, supplied: Mapping[Literal["ri", "ir", "ii"], Path] | None
) -> Spectra:
    """Stack identified component planes without treating them as independent series."""
    if real.n_spectral_dims != _QUADRATURE_DIMENSIONS or real.data.shape[0] != 1:
        raise ValueError("--use-imaginary requires one phase-sensitive 2D spectrum.")
    paths = {str(key): value for key, value in (supplied or {}).items()}
    if set(paths) - {"ri", "ir", "ii"}:
        raise ValueError("Imaginary component names must be ri, ir, or ii.")
    if not paths and resolved_path.is_dir():
        paths = {
            component: resolved_path / f"2{component}"
            for component in ("ri", "ir", "ii")
            if (resolved_path / f"2{component}").is_file()
        }
    if not paths:
        raise ValueError(
            "--use-imaginary requires at least one imaginary component. "
            "Place 2ri/2ir/2ii beside 2rr, or supply --imaginary ri=PATH, ir=PATH, or ii=PATH."
        )
    components = ["rr"]
    planes = [real.data[0]]
    used_paths = {"rr": str(resolved_path / "2rr" if resolved_path.is_dir() else resolved_path)}
    for component in ("ri", "ir", "ii"):
        if component not in paths:
            continue
        path = Path(paths[component]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Imaginary component {component} is not a file: {path}")
        if (path.parent / "procs").is_file():
            dic, data, params, added = _read_bruker(path.parent, path.name)
            other = Spectra(
                dic=dic, data=data, params=params, z_values=np.array([0.0]), pseudo_dim_added=added
            )
        else:
            other = read_spectra(path)
        _validate_component_grid(real, other, component)
        components.append(component)
        planes.append(other.data[0])
        used_paths[component] = str(path)
    dic = dict(real.dic)
    dic["_peakfit_quadrature_paths"] = used_paths
    params = [real.params[0].model_copy(update={"size": len(components)}), *real.spectral_params]
    return Spectra(
        dic=dic,
        data=np.stack(planes),
        params=params,
        z_values=np.arange(len(components), dtype=float),
        pseudo_dim_added=True,
        quadrature_components=tuple(components),
    )


def _validate_component_grid(real: Spectra, other: Spectra, component: str) -> None:
    if other.data.shape != real.data.shape or other.n_spectral_dims != _QUADRATURE_DIMENSIONS:
        raise ValueError(f"Imaginary component {component} must match the real 2D grid shape.")
    for expected, actual in zip(real.spectral_params, other.spectral_params, strict=True):
        geometry = ("size", "sw", "obs", "car")
        if any(
            not np.isclose(getattr(expected, field), getattr(actual, field), rtol=1e-8, atol=1e-8)
            for field in geometry
        ):
            raise ValueError(f"Imaginary component {component} has a different spectral grid.")
        processing = (
            "aq_time",
            "apocode",
            "apodq1",
            "apodq2",
            "apodq3",
            "p180",
            "td_size",
            "ft_size",
            "dqfcosy_window",
        )
        if any(getattr(expected, field) != getattr(actual, field) for field in processing):
            raise ValueError(f"Imaginary component {component} has different processing or timing.")


__all__ = [
    "BRUKER_BINARY_NAMES",
    "NUCLEUS_LABELS",
    "read_spectra",
    "read_spectral_parameters",
    "resolve_spectrum_path",
]
