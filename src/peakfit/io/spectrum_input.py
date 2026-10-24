"""Validated external spectrum coordinates and all-axis region extraction.

No fitting, terminal, or GUI dependencies belong in this module.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from nmrglue.fileio import bruker, pipe, sparky
from pydantic import BaseModel, ConfigDict, Field

from peakfit.engine.domain.config import DQFCOSYWindow
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, get_dimension_label
from peakfit.engine.lineshapes.dqfcosy import _header_window
from peakfit.io.readers.spectrum import (
    P1_MAX,
    P1_MIN,
    _bruker_axis_key,
    _pseudo_parameters,
    _read_bruker,
    resolve_spectrum_path,
)

# Absolute tolerance in ppm, used only to absorb floating-point endpoint error.
PPM_TOLERANCE = 1e-9
_MAX_FREQUENCY_AXES = 3
_BOUND_COUNT = 2
_MIN_PROCESSING_POINTS = 2
_NMRPIPE_DIRECT_DIMENSION = 2
_PLACEHOLDERS = {"", "x", "y", "z", "a", "unknown", "none"}
type Region = Mapping[int, tuple[float, float]]
type SeriesUnit = Literal["s", "ms", "us", "Hz", "ppm", "index"]
type SeriesKind = Literal["time", "relaxation", "cest", "replicate"]


class AxisProcessing(BaseModel):
    """Separately sourced acquisition/processing history, never derived from a crop."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    provenance: str = Field(min_length=1)
    acquisition_time: float | None = Field(default=None, gt=0)
    td_size: int | None = Field(default=None, ge=2)
    ft_size: int | None = Field(default=None, ge=2)
    apod_code: int | None = None
    apod_q1: float | None = None
    apod_q2: float | None = None
    apod_q3: float | None = None
    phase0_deg: float | None = None
    phase1_deg: float | None = None
    dqfcosy_window: DQFCOSYWindow | None = None


@dataclass(frozen=True)
class SeriesAxis:
    """Ordered series coordinates and optional replicate identities; no sorting."""

    values: tuple[float, ...]
    unit: SeriesUnit
    kind: SeriesKind
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Detach values and validate units and stable member identities."""
        values = tuple(float(value) for value in self.values)
        if not values or not np.isfinite(values).all():
            raise ValueError("Series values must be finite and nonempty.")
        units = {
            "time": {"s", "ms", "us"},
            "relaxation": {"s", "ms", "us"},
            "cest": {"Hz", "ppm"},
            "replicate": {"index"},
        }
        if self.kind not in units or self.unit not in units[self.kind]:
            raise ValueError("Series kind requires explicit compatible units.")
        ids = tuple(self.ids) or tuple(str(i) for i in range(len(values)))
        if len(ids) != len(values) or len(set(ids)) != len(ids) or any(not x for x in ids):
            raise ValueError("Series ids must be nonempty, unique, and match the values.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "ids", ids)


def freeze_metadata(value: Any) -> Any:
    """Detach nested metadata into read-only containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    return deepcopy(value)


@dataclass(frozen=True)
class SpectrumInput:
    """External-axis spectrum descriptor. Arrays are borrowed until extraction.

    Metadata is detached immediately. Extraction copies only the selected data.
    Do not modify a borrowed array concurrently with preparation.
    """

    data: Any = field(repr=False)
    udic: Mapping[Any, Any]
    direct_axis: int
    series_axis: int | None = None
    series: SeriesAxis | None = None
    processing: Mapping[int, AxisProcessing] = field(default_factory=dict)
    native_headers: Mapping[str, Any] = field(default_factory=dict, repr=False)
    source_format: str = "array"
    loading: str = "borrowed array; selected region copied"
    source_path: str | None = None

    def __post_init__(self) -> None:
        """Validate dimension identities before data allocation."""
        object.__setattr__(self, "udic", freeze_metadata(self.udic))
        object.__setattr__(self, "native_headers", freeze_metadata(self.native_headers))
        object.__setattr__(self, "processing", MappingProxyType(dict(self.processing)))
        _validate_input(self)

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        udic: Mapping[Any, Any],
        *,
        direct_axis: int,
        series_axis: int | None = None,
        series: SeriesAxis | None = None,
        processing: Mapping[int, AxisProcessing] | None = None,
    ) -> SpectrumInput:
        """Accept real processed ndarray data using standard nmrglue UDIC axes."""
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a real numeric NumPy ndarray.")
        return cls(data, udic, direct_axis, series_axis, series, processing or {})

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        direct_axis: int | None = None,
        series_axis: int | None = None,
        series: SeriesAxis | None = None,
        processing: Mapping[int, AxisProcessing] | None = None,
    ) -> SpectrumInput:
        """Read native coordinates; defer NMRPipe/UCSF data reads where supported."""
        return _file_input(Path(path), direct_axis, series_axis, series, processing or {})

    @property
    def frequency_axes(self) -> tuple[int, ...]:
        """External frequency axes, reordered with the acquisition axis last."""
        return (
            *tuple(
                i for i in range(self.data.ndim) if i not in {self.series_axis, self.direct_axis}
            ),
            self.direct_axis,
        )


def _validate_input(source: SpectrumInput, *, intensities: bool = True) -> None:
    data, udic = source.data, source.udic
    if np.dtype(data.dtype).kind not in "iuf" or any(size < 1 for size in data.shape):
        raise ValueError("Spectrum must contain real numeric intensities and nonempty axes.")
    if (
        type(udic.get("ndim")) is not int
        or udic.get("ndim") != data.ndim
        or {k for k in udic if isinstance(k, int)} != set(range(data.ndim))
    ):
        raise ValueError("UDIC ndim and integer axis keys must match the ndarray axes exactly.")
    if type(source.direct_axis) is not int or not 0 <= source.direct_axis < data.ndim:
        raise ValueError("direct_axis must identify an external frequency axis.")
    if source.series_axis is not None and (
        type(source.series_axis) is not int
        or not 0 <= source.series_axis < data.ndim
        or source.series_axis == source.direct_axis
    ):
        raise ValueError("series_axis must be a different valid external axis.")
    if (source.series_axis is None) != (source.series is None):
        raise ValueError("A series axis requires SeriesAxis values with explicit units.")
    if source.series is not None and len(source.series.values) != data.shape[source.series_axis]:
        raise ValueError("Series values must have exactly the series axis length.")
    if not 1 <= len(source.frequency_axes) <= _MAX_FREQUENCY_AXES:
        raise ValueError(
            "Support requires one to three frequencies; "
            "identify an explicit series_axis for series data."
        )
    if set(source.processing) - set(source.frequency_axes) or any(
        not isinstance(item, AxisProcessing) for item in source.processing.values()
    ):
        raise ValueError("processing must map external frequency axes to AxisProcessing objects.")
    for axis, size in enumerate(data.shape):
        _validate_axis(udic[axis], axis, size, axis == source.series_axis)
    if intensities and isinstance(data, np.ndarray):
        # Bounded validation even for large noncontiguous/read-only inputs.
        for block in np.nditer(data, flags=["external_loop", "buffered"], buffersize=131072):
            if not np.isfinite(block).all():
                raise ValueError("Spectrum contains nonfinite intensities.")


def _validate_axis(meta: Mapping[str, Any], axis: int, size: int, is_series: bool) -> None:
    if type(meta.get("size")) is not int or meta["size"] != size:
        raise ValueError(f"UDIC axis {axis} size does not match data shape.")
    for flag in ("freq", "time", "complex"):
        if not isinstance(meta.get(flag), (bool, np.bool_)):
            raise ValueError(f"UDIC axis {axis} requires an explicit boolean {flag}.")
    if meta["complex"]:
        raise ValueError(f"UDIC axis {axis}: complex/hypercomplex layouts are unsupported here.")
    if not isinstance(meta.get("label"), str):
        raise ValueError(f"UDIC axis {axis} requires a label.")
    if is_series:
        if meta["freq"]:
            raise ValueError("A series axis must not be marked as a frequency dimension.")
        return
    if not meta["freq"] or meta["time"]:
        raise ValueError(
            f"Axis {axis} requires frequency data, not time-domain or an implicit series."
        )
    if meta["label"].strip().lower() in _PLACEHOLDERS:
        raise ValueError(f"Axis {axis} label is missing or a generic placeholder.")
    for key in ("sw", "obs", "car"):
        value = meta.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            raise ValueError(f"UDIC axis {axis} requires finite {key} metadata.")
        if key != "car" and value <= 0:
            raise ValueError(
                f"UDIC axis {axis} {key} must be positive; ascending grids are unsupported."
            )


@dataclass(frozen=True)
class AxisExtraction:
    """One frequency-axis selection in external and internal coordinates."""

    external_axis: int
    internal_axis: int
    label: str
    requested_ppm: tuple[float, float]
    sampled_ppm: tuple[float, float]
    fitting_ppm: tuple[float, float]
    analysis_slice: slice
    fitting_slice: slice
    halo_clipped: bool
    neighbor_expanded: bool = False


@dataclass(frozen=True)
class ResourceEstimate:
    """Conservative array/workspace budget, not an operating-system memory guarantee."""

    selected_points: int
    data_bytes: int
    worker_overhead_bytes: int
    output_bytes: int
    estimated_peak_bytes: int
    workers: int


@dataclass(frozen=True)
class Extraction:
    """Detached crop provenance. Native headers describe the source, not this grid."""

    original_shape: tuple[int, ...]
    original_slices: tuple[slice, ...]
    axis_permutation: tuple[int, ...]
    axes: tuple[AxisExtraction, ...]
    cropped_udic: Mapping[Any, Any]
    native_headers: Mapping[str, Any]
    processing: Mapping[int, AxisProcessing]
    series: SeriesAxis | None
    source_format: str
    loading: str
    diagnostics: tuple[str, ...]
    resources: ResourceEstimate
    series_indices: tuple[int, ...] = ()
    source_path: str | None = None


@dataclass(frozen=True)
class ExtractedSpectrum:
    """Prepared domain spectrum and provenance, usable independently of fitting."""

    spectra: Spectra
    extraction: Extraction


def _bounds_slice(
    meta: Mapping[str, Any], bounds: tuple[float, float]
) -> tuple[slice, tuple[float, float]]:
    if len(bounds) != _BOUND_COUNT or not np.isfinite(bounds).all():
        raise ValueError("Region bounds must be two finite ppm values.")
    low, high = sorted(float(value) for value in bounds)
    size, sw, obs, car = (meta[key] for key in ("size", "sw", "obs", "car"))
    delta = -sw / (size * obs)
    first = car / obs + sw / (2 * obs)
    last = first + (size - 1) * delta
    if low < last - PPM_TOLERANCE or high > first + PPM_TOLERANCE:
        raise ValueError(
            "Region is outside the stored spectrum; aliased/folded regions are unsupported."
        )
    # Inclusive sampled interval, without nearest-point rounding or modulo wrapping.
    start = max(0, int(np.ceil((first - high - PPM_TOLERANCE) / -delta)))
    stop = min(size, int(np.floor((first - low + PPM_TOLERANCE) / -delta)) + 1)
    if start >= stop:
        raise ValueError("Region contains no sampled points.")
    return slice(start, stop), (low, high)


def _select_series(source: SpectrumInput, exclude_planes: Sequence[int]):
    series = source.series
    count = len(series.values) if series else 1
    if any(type(i) is not int or not 0 <= i < count for i in exclude_planes):
        raise ValueError("exclude_planes must contain valid original series indices.")
    series_indices = tuple(i for i in range(count) if i not in exclude_planes)
    if not series_indices:
        raise ValueError("exclude_planes leaves no series members to fit.")
    if series is not None:
        series = SeriesAxis(
            tuple(series.values[i] for i in series_indices),
            series.unit,
            series.kind,
            tuple(series.ids[i] for i in series_indices),
        )
    return series, series_indices


def plan_extraction(
    source: SpectrumInput,
    region_ppm: Region | None = None,
    *,
    halo_points: int = 0,
    workers: int = 1,
    peak_count: int = 1,
    max_memory_bytes: int | None = None,
    exclude_planes: Sequence[int] = (),
    _include_centers: tuple[Mapping[int, float], ...] = (),
) -> Extraction:
    """Plan all-axis slices and a resource budget before allocating spectrum data."""
    _validate_input(source, intensities=False)
    if type(halo_points) is not int or halo_points < 0:
        raise ValueError("halo_points must be a nonnegative integer.")
    if type(workers) is not int or type(peak_count) is not int or workers < 1 or peak_count < 1:
        raise ValueError("workers and peak_count must be positive.")
    axes = source.frequency_axes
    if region_ppm is not None and (
        any(type(k) is not int for k in region_ppm) or set(region_ppm) != set(axes)
    ):
        raise ValueError(
            "region_ppm requires every external frequency axis and excludes the series axis."
        )
    slices = [slice(0, size) for size in source.data.shape]
    selections = []
    cropped = {"ndim": source.data.ndim}
    for internal, axis in enumerate(axes, start=1):
        meta = dict(source.udic[axis])
        size, sw, obs, car = (meta[key] for key in ("size", "sw", "obs", "car"))
        delta, first = -sw / (size * obs), car / obs + sw / (2 * obs)
        bounds = region_ppm[axis] if region_ppm is not None else (first, first + (size - 1) * delta)
        analysis, requested = _bounds_slice(meta, bounds)
        start, stop = max(0, analysis.start - halo_points), min(size, analysis.stop + halo_points)
        clipped = start != analysis.start - halo_points or stop != analysis.stop + halo_points
        original_fit = (start, stop)
        for center in _include_centers:
            point = (center[axis] - first) / delta
            start = min(start, max(0, int(np.floor(point + PPM_TOLERANCE / abs(delta)))))
            stop = max(stop, min(size, int(np.ceil(point - PPM_TOLERANCE / abs(delta))) + 1))
        fit_slice = slice(start, stop)
        slices[axis] = fit_slice
        new_size = stop - start
        new_sw = sw * new_size / size
        new_first = first + start * delta
        meta.update(size=new_size, sw=new_sw, car=new_first * obs - new_sw / 2)
        cropped[axis] = meta
        selections.append(
            AxisExtraction(
                axis,
                internal,
                get_dimension_label(internal - 1),
                requested,
                (first + analysis.start * delta, first + (analysis.stop - 1) * delta),
                (new_first, first + (stop - 1) * delta),
                analysis,
                fit_slice,
                clipped,
                original_fit != (start, stop),
            )
        )
    if source.series_axis is not None:
        cropped[source.series_axis] = dict(source.udic[source.series_axis])
    series, series_indices = _select_series(source, exclude_planes)
    if series is not None:
        cropped[source.series_axis]["size"] = len(series_indices)
    shape = tuple(
        (len(series_indices) if axis == source.series_axis else s.stop - s.start)
        for axis, s in enumerate(slices)
    )
    points = int(np.prod(shape, dtype=np.int64))
    dtype = np.dtype(source.data.dtype)
    itemsize = dtype.itemsize if dtype.kind == "f" else 8
    data_bytes = points * itemsize
    permutation = ((source.series_axis,) if source.series_axis is not None else ()) + axes
    # Data, float64 clustering/corrections/Jacobian, final evaluations, and optional simulation.
    overhead = points * 8 * (12 + 4 * peak_count * len(axes)) * workers
    output_bytes = points * 16
    estimate = ResourceEstimate(
        points, data_bytes, overhead, output_bytes, data_bytes + overhead + output_bytes, workers
    )
    if max_memory_bytes is not None and (
        max_memory_bytes < 1 or estimate.estimated_peak_bytes > max_memory_bytes
    ):
        raise MemoryError(
            f"Estimated fitting allocation {estimate.estimated_peak_bytes} "
            f"exceeds max_memory_bytes={max_memory_bytes}."
        )
    diagnostics = [
        "Crop edges are not folding boundaries; indirect lineshapes retain the source period."
    ]
    diagnostics.extend(
        f"Axis {x.external_axis}: fitting extent expanded to include a neighboring seed."
        for x in selections
        if x.neighbor_expanded
    )
    diagnostics.extend(
        f"Axis {x.external_axis}: halo clipped at stored boundary."
        for x in selections
        if x.halo_clipped
    )
    return Extraction(
        tuple(source.data.shape),
        tuple(slices),
        permutation,
        tuple(selections),
        freeze_metadata(cropped),
        source.native_headers,
        source.processing,
        series,
        source.source_format,
        source.loading,
        tuple(diagnostics),
        estimate,
        series_indices,
        source.source_path,
    )


def extract_spectrum(
    source: SpectrumInput,
    region_ppm: Region | None = None,
    *,
    halo_points: int = 0,
    workers: int = 1,
    peak_count: int = 1,
    max_memory_bytes: int | None = None,
    exclude_planes: Sequence[int] = (),
    _include_centers: tuple[Mapping[int, float], ...] = (),
) -> ExtractedSpectrum:
    """Select a closed ppm interval on every frequency axis, then copy that crop."""
    extraction = plan_extraction(
        source,
        region_ppm,
        halo_points=halo_points,
        workers=workers,
        peak_count=peak_count,
        max_memory_bytes=max_memory_bytes,
        _include_centers=_include_centers,
        exclude_planes=exclude_planes,
    )
    slices = extraction.original_slices
    permutation = extraction.axis_permutation
    cropped = extraction.cropped_udic
    selections = extraction.axes
    dtype = np.dtype(source.data.dtype)
    selected = np.asarray(source.data[tuple(slices)])
    if not np.isfinite(selected).all():
        raise ValueError("Selected spectrum contains nonfinite intensities.")
    selected = selected.transpose(permutation)
    if source.series_axis is not None and exclude_planes:
        selected = selected[list(extraction.series_indices)]
    if source.series_axis is None:
        selected = selected[None, ...]
    data = np.array(selected, dtype=None if dtype.kind == "f" else np.float64, order="C", copy=True)
    data.flags.writeable = False
    params = [_pseudo_parameters(data.shape[0])]
    for selection in selections:
        axis, label = selection.external_axis, selection.label
        meta = cropped[axis]
        proc = source.processing.get(axis)
        params.append(
            SpectralParameters(
                size=meta["size"],
                sw=meta["sw"],
                obs=meta["obs"],
                car=meta["car"],
                aq_time=proc.acquisition_time
                if proc and proc.acquisition_time is not None
                else 0.0,
                apocode=proc.apod_code if proc and proc.apod_code is not None else -1.0,
                apodq1=proc.apod_q1 if proc and proc.apod_q1 is not None else 0.0,
                apodq2=proc.apod_q2 if proc and proc.apod_q2 is not None else 0.0,
                apodq3=proc.apod_q3 if proc and proc.apod_q3 is not None else 0.0,
                p180=bool(
                    proc
                    and proc.phase1_deg is not None
                    and P1_MIN <= abs(proc.phase1_deg) <= P1_MAX
                ),
                direct=axis == source.direct_axis,
                ft=True,
                label=label,
                nucleus=meta["label"],
                td_size=proc.td_size if proc else None,
                ft_size=proc.ft_size if proc else None,
                dqfcosy_window=proc.dqfcosy_window if proc else None,
                alias_size=source.udic[axis]["size"],
            )
        )
    values = np.array(extraction.series.values if extraction.series else [0.0], dtype=np.float64)
    values.flags.writeable = False
    spectra = Spectra(
        dic={"_peakfit_format": source.source_format},
        data=data,
        z_values=values,
        params=params,
        pseudo_dim_added=source.series_axis is None,
    )
    return ExtractedSpectrum(spectra, extraction)


def _bruker_input_parts(resolved: Path, series_axis: int | None):
    processing = {}
    native, internal_data, params, added = _read_bruker(resolved)
    data = internal_data[0] if added else internal_data
    udic = bruker.guess_udic(native, data, strip_fake=True)
    for axis in range(data.ndim):
        proc = native.get(_bruker_axis_key("proc", data.ndim, axis), {})
        if "FT_mod" not in proc:
            raise ValueError(f"Bruker axis {axis} is missing processed-domain FT_mod.")
        # guess_udic describes acquisition quadrature even after read_pdata.
        # The explicitly selected 1r/2rr/3rrr binary contains real components.
        frequency = proc["FT_mod"] != 0
        udic[axis].update(freq=frequency, time=not frequency, complex=False)
        udic[axis]["label"] = udic[axis]["label"].strip("<> ")
    for axis in range(data.ndim):
        if axis == series_axis:
            continue
        proc = native.get(_bruker_axis_key("proc", data.ndim, axis), {})
        acqu = native.get(_bruker_axis_key("acqu", data.ndim, axis), {})
        # Only offer native processing models when the reader obtained full history.
        index = axis + 1 if added else axis
        param = params[index]
        complete = (
            {"WDW", "PHC0", "PHC1"} <= proc.keys()
            and "TD" in acqu
            and ("SW_h" in acqu or {"SW", "SFO1"} <= acqu.keys())
        )
        if complete:
            processing[axis] = AxisProcessing(
                provenance="Bruker acquisition and processing headers",
                acquisition_time=param.aq_time if param.aq_time > 0 else None,
                td_size=param.td_size,
                ft_size=param.ft_size,
                apod_code=int(param.apocode),
                apod_q1=param.apodq1,
                apod_q2=param.apodq2,
                apod_q3=param.apodq3,
                phase0_deg=proc["PHC0"],
                phase1_deg=proc["PHC1"],
                dqfcosy_window=param.dqfcosy_window,
            )
    return native, data, udic, processing


def _pipe_window(processing: AxisProcessing, meta: Mapping[str, Any]) -> DQFCOSYWindow | None:
    """Use the existing native-window validator before the frequency grid is cropped."""
    aq, code = processing.acquisition_time, processing.apod_code
    q1, q2, q3 = processing.apod_q1, processing.apod_q2, processing.apod_q3
    if aq is None or code is None or q1 is None or q2 is None or q3 is None:
        return None
    params = SpectralParameters(
        size=meta["size"],
        sw=meta["sw"],
        obs=meta["obs"],
        car=meta["car"],
        aq_time=aq,
        apocode=code,
        apodq1=q1,
        apodq2=q2,
        apodq3=q3,
        p180=False,
        direct=False,
        ft=True,
        ft_size=processing.ft_size,
    )
    try:
        return _header_window(params)
    except ValueError:
        return None


def _file_input(path, direct_axis, series_axis, series, extra) -> SpectrumInput:
    resolved = resolve_spectrum_path(path)
    processing = {}
    if resolved.is_dir():
        native, data, udic, processing = _bruker_input_parts(resolved, series_axis)
        fmt, loading = "bruker", "full processed Bruker loading before extraction"
        if direct_axis is None:
            direct_axis = data.ndim - 1
    elif resolved.suffix.lower() in {".ucsf", ".sparky"}:
        native, data = sparky.read_lowmem(str(resolved))
        udic = sparky.guess_udic(native, data)
        fmt, loading = "ucsf", "bounded UCSF access through nmrglue"
        if direct_axis is None:
            raise ValueError(
                "UCSF headers do not establish the acquisition axis; supply direct_axis."
            )
    else:
        native, data = pipe.read_lowmem(str(resolved))
        udic = pipe.guess_udic(native, data)
        # CAR may refer to the original grid after NMRPipe EXT. ORIG is the stored edge.
        for axis in range(data.ndim):
            if axis == series_axis:
                continue
            fn = f"FDF{int(native['FDDIMORDER'][data.ndim - 1 - axis])}"
            meta = udic[axis]
            meta["car"] = native[fn + "ORIG"] + meta["sw"] / 2 - meta["sw"] / meta["size"]
            _validate_axis(meta, axis, data.shape[axis], False)
            keys = [
                "APOD",
                "TDSIZE",
                "FTSIZE",
                "APODCODE",
                "APODQ1",
                "APODQ2",
                "APODQ3",
                "P0",
                "P1",
            ]
            # An EXT header can describe a cropped grid while APOD still counts
            # original acquisition points. SW then need not be the acquisition
            # width, so APOD/SW cannot establish acquisition duration.
            extracted_header = native.get(fn + "X1", 0) != 0 or native.get(fn + "XN", 0) != 0
            if (
                not extracted_header
                and all(fn + key in native for key in keys)
                and np.isfinite([native[fn + key] for key in keys]).all()
                and native[fn + "FTSIZE"] == meta["size"]
                and native[fn + "APOD"] > 0
                and native[fn + "TDSIZE"] >= _MIN_PROCESSING_POINTS
                and native[fn + "FTSIZE"] >= _MIN_PROCESSING_POINTS
            ):
                processing[axis] = AxisProcessing(
                    provenance="NMRPipe processing headers",
                    acquisition_time=native[fn + "APOD"] / meta["sw"],
                    td_size=int(native[fn + "TDSIZE"]) or None,
                    ft_size=int(native[fn + "FTSIZE"]) or None,
                    apod_code=int(native[fn + "APODCODE"]),
                    apod_q1=native[fn + "APODQ1"],
                    apod_q2=native[fn + "APODQ2"],
                    apod_q3=native[fn + "APODQ3"],
                    phase0_deg=native[fn + "P0"],
                    phase1_deg=native[fn + "P1"],
                )
                processing[axis] = processing[axis].model_copy(
                    update={
                        "dqfcosy_window": _pipe_window(processing[axis], meta),
                    }
                )
        if direct_axis is None:
            direct_axis = next(
                axis
                for axis in range(data.ndim)
                if int(native["FDDIMORDER"][data.ndim - 1 - axis]) == _NMRPIPE_DIRECT_DIMENSION
            )
        fmt = "nmrpipe"
        loading = (
            "full 1D NMRPipe loading"
            if data.ndim == 1
            else "bounded NMRPipe access through nmrglue"
        )
    processing.update(extra)
    return SpectrumInput(
        data,
        udic,
        direct_axis,
        series_axis,
        series,
        processing,
        native,
        fmt,
        loading,
        source_path=str(resolved),
    )


def read_extracted_spectrum(source_path: Path, extraction_path: Path) -> Spectra:
    """Reload a saved region from its source, without fitting or continuation state.

    The stored slices select samples; stored frequency metadata must agree with
    the native source at those exact samples before any model can be displayed.
    """
    import json  # noqa: PLC0415

    from peakfit.io.readers.spectrum import read_spectra  # noqa: PLC0415

    saved = json.loads(extraction_path.read_text())["extraction"]
    original = read_spectra(source_path)
    raw = original.data[0] if original.pseudo_dim_added else original.data
    if tuple(raw.shape) != tuple(saved["original_shape"]):
        raise ValueError("Source spectrum shape does not match the saved extraction.")
    permutation = tuple(saved["axis_permutation"])
    if sorted(permutation) != list(range(raw.ndim)):
        raise ValueError("Saved extraction has an invalid axis permutation.")
    slices = tuple(slice(item["start"], item["stop"]) for item in saved["original_slices"])
    if any(item.get("step") not in {None, 1} for item in saved["original_slices"]):
        raise ValueError("Saved extraction source slices must have unit stride.")
    if len(slices) != raw.ndim or any(
        item.start < 0 or item.stop > size or item.start >= item.stop
        for item, size in zip(slices, raw.shape, strict=True)
    ):
        raise ValueError("Saved extraction contains invalid source slices.")
    udic = {
        int(key) if key != "ndim" else key: value for key, value in saved["cropped_udic"].items()
    }
    for axis in saved["axes"]:
        external = axis["external_axis"]
        native_param = original.params[external + int(original.pseudo_dim_added)]
        meta = udic[external]
        native_indices = np.array([slices[external].start, slices[external].stop - 1], dtype=float)
        first = meta["car"] / meta["obs"] + meta["sw"] / (2 * meta["obs"])
        last = first - (meta["size"] - 1) * meta["sw"] / (meta["size"] * meta["obs"])
        if native_param.nucleus != meta["label"] or not np.allclose(
            native_param.pts2ppm(native_indices), [first, last], rtol=0, atol=PPM_TOLERANCE
        ):
            raise ValueError(f"Source frequency grid does not match saved axis {external}.")
    external_axes = {axis["external_axis"] for axis in saved["axes"]}
    remaining = set(range(raw.ndim)) - external_axes
    if len(remaining) > 1:
        raise ValueError("Saved extraction has more than one series axis.")
    series_axis = next(iter(remaining)) if remaining else None
    series = SeriesAxis(**saved["series"]) if saved["series"] is not None else None
    selected = raw[slices]
    if series_axis is not None:
        selected = np.take(selected, saved["series_indices"], axis=series_axis)
    processing = {
        int(axis): AxisProcessing.model_validate(values)
        for axis, values in saved["processing"].items()
    }
    source = SpectrumInput.from_array(
        selected,
        udic,
        direct_axis=permutation[-1],
        series_axis=series_axis,
        series=series,
        processing=processing,
    )
    extracted = extract_spectrum(source)
    for axis, param in zip(saved["axes"], extracted.spectra.spectral_params, strict=True):
        param.alias_size = raw.shape[axis["external_axis"]]
    extracted.spectra.dic["_peakfit_format"] = saved["source_format"]
    return extracted.spectra
