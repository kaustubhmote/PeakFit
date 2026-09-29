"""Phase-sensitive, weakly coupled DQF-COSY cross-peaks.

Each axis is a difference of two windowed, exponentially relaxing resonances.
Their product gives the four signed lobes. Both axes use one shared positive J.
The finite geometric sums reproduce a sampled FID without evaluating an FFT
inside the optimizer. Window inputs remain fixed while intrinsic widths vary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from peakfit.engine.domain.config import DQFCOSYWindow
from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.lineshapes.templates import SimpleSingletBase
from peakfit.engine.lineshapes.utils import estimate_cs_bounds_ppm
from peakfit.engine.types import LineshapeResult, ParamSpec

_TAYLOR_RADIUS = 1e-3
_SPECTRAL_DIMENSIONS = 2
_MIN_TIME_POINTS = 2

if TYPE_CHECKING:
    from peakfit.engine.domain.config import FitConfig
    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
    from peakfit.engine.lineshapes.utils import LineshapeContext
    from peakfit.engine.types import ClusterParameters
    from peakfit.shared.typing import ComplexArray, FloatArray


def _finite_integral(z: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
    """Return (1-exp(-z))/z and its derivative, including the removable pole."""
    small = np.abs(z) < _TAYLOR_RADIUS
    safe = np.where(small, 1.0 + 0j, z)
    exponential = np.exp(-safe)
    value = -np.expm1(-safe) / safe
    derivative = (exponential * (safe + 1.0) - 1.0) / safe**2
    # Taylor series through z**5, also valid when a shifted window hits z=0.
    value = np.where(
        small, 1 + z * (-1 / 2 + z * (1 / 6 + z * (-1 / 24 + z * (1 / 120 - z / 720)))), value
    )
    derivative = np.where(
        small, -1 / 2 + z * (1 / 3 + z * (-1 / 8 + z * (1 / 30 - z / 144))), derivative
    )
    return value, derivative


def _geometric_mean(z: ComplexArray, points: int) -> tuple[ComplexArray, ComplexArray]:
    """Mean exp(-n*z) and derivative for n=0,...,points-1."""
    # The sampled transform is periodic, including at exact sweep-width offsets.
    z = z.real + 1j * ((z.imag + np.pi) % (2 * np.pi) - np.pi)
    numerator, numerator_derivative = _finite_integral(points * z)
    denominator, denominator_derivative = _finite_integral(z)
    value = numerator / denominator
    derivative = (points * numerator_derivative - value * denominator_derivative) / denominator
    return value, derivative


def windowed_resonance(
    offsets: FloatArray, linewidth: FloatArray, window: DQFCOSYWindow
) -> tuple[ComplexArray, dict[str, ComplexArray]]:
    """Transform exp(-pi*lw*t)*window(t), normalized by the acquired point count.

    Width and offsets are in Hz. t[n]=n*acquisition_time/points. A sine window
    includes both endpoints, matching NMRPipe SP. Exponential broadening adds
    pi*lb_hz to the transverse relaxation rate, R2=pi*lw.
    """
    dwell = window.acquisition_time / window.points
    z = np.pi * dwell * (linewidth + window.lb_hz + 2j * offsets)
    alpha = np.pi * window.sine_offset
    beta = np.pi * (window.sine_end - window.sine_offset) / (window.points - 1)
    if window.sine_power == 0:
        terms = [(1.0 + 0j, 0.0)]
    elif window.sine_power == 1:
        terms = [(-0.5j * np.exp(1j * alpha), beta), (0.5j * np.exp(-1j * alpha), -beta)]
    else:
        terms = [
            (0.5 + 0j, 0.0),
            (-0.25 * np.exp(2j * alpha), 2 * beta),
            (-0.25 * np.exp(-2j * alpha), -2 * beta),
        ]
    value = np.zeros_like(z, dtype=np.complex128)
    derivative = np.zeros_like(z, dtype=np.complex128)
    for weight, shift in terms:
        component, component_derivative = _geometric_mean(z - 1j * shift, window.points)
        value += weight * component
        derivative += weight * component_derivative
    return value, {"lw": np.pi * dwell * derivative, "dw": 2j * np.pi * dwell * derivative}


def antiphase_profile(
    offsets: FloatArray, linewidth: FloatArray, coupling: FloatArray, window: DQFCOSYWindow
) -> tuple[ComplexArray, dict[str, ComplexArray]]:
    """Return the signed doublet and derivatives with respect to Hz parameters."""
    left, d_left = windowed_resonance(offsets - coupling / 2, linewidth, window)
    right, d_right = windowed_resonance(offsets + coupling / 2, linewidth, window)
    return left - right, {
        "dw": d_left["dw"] - d_right["dw"],
        "lw": d_left["lw"] - d_right["lw"],
        "j": -0.5 * (d_left["dw"] + d_right["dw"]),
    }


def _header_window(spec: SpectralParameters) -> DQFCOSYWindow:
    """Read supported NMRPipe windows without substituting an empirical shape."""
    if spec.ft_size is not None and spec.ft_size != spec.size:
        raise ValueError(
            f"DQF-COSY: {spec.label} was extracted or resized after Fourier transformation. "
            f"Set fitting.dqfcosy_windows.{spec.label} using the original acquisition timing."
        )
    sine_code, exponential_code = 1.0, 2.0
    if spec.apocode not in {0.0, sine_code, exponential_code} or (
        spec.apocode == sine_code and spec.apodq3 not in {1.0, 2.0}
    ):
        raise ValueError(
            f"DQF-COSY: unsupported or unknown window on {spec.label}. "
            f"Set fitting.dqfcosy_windows.{spec.label} to the actual processing settings."
        )
    points = spec.aq_time * spec.sw
    if not np.isfinite(points) or points < _MIN_TIME_POINTS:
        raise ValueError(
            f"DQF-COSY: missing acquisition time on {spec.label}; "
            f"set fitting.dqfcosy_windows.{spec.label}."
        )
    return DQFCOSYWindow(
        acquisition_time=spec.aq_time,
        points=round(points),
        sine_power=(1 if spec.apodq3 == 1.0 else 2) if spec.apocode == sine_code else 0,
        sine_offset=spec.apodq1 if spec.apocode == sine_code else 0.0,
        sine_end=spec.apodq2 if spec.apocode == sine_code else 1.0,
        lb_hz=spec.apodq1 if spec.apocode == exponential_code else 0.0,
    )


def param_specs(center: float, context: LineshapeContext | None = None) -> tuple[ParamSpec, ...]:
    """Fit multiplet midpoint, intrinsic FWHM, shared J, and optional phase."""
    bounds = estimate_cs_bounds_ppm(context)
    return (
        ParamSpec("cs", center, center - bounds, center + bounds, unit="ppm"),
        ParamSpec("lw", 6.0, 0.1, 100.0, unit="Hz"),
        ParamSpec("j", 7.0, 0.1, 20.0, unit="Hz"),
        ParamSpec("phase", 0.0, -15.0, 15.0, unit="deg"),
    )


class DQFCOSY(SimpleSingletBase):
    """One axis of a four-lobe cross-peak, with one J shared by both axes."""

    shape_name: ClassVar[str] = "dqfcosy"
    param_specs = staticmethod(param_specs)

    def __init__(
        self, name: str, center: float, spectra: Spectra, dim: int, config: FitConfig
    ) -> None:
        super().__init__(name, center, spectra, dim, config)
        if spectra.n_spectral_dims != _SPECTRAL_DIMENSIONS or config.tilted:
            raise ValueError("DQF-COSY requires two spectral dimensions and tilted=false.")
        axes = [p.label or f"F{i + 2}" for i, p in enumerate(spectra.spectral_params)]
        unknown = set(config.dqfcosy_windows) - set(axes)
        if unknown:
            raise ValueError(f"Unknown DQF-COSY window axes: {sorted(unknown)}; use {axes}.")
        self._coupling_axis = "-".join(axes)
        window = config.dqfcosy_windows.get(self.axis)
        if window is None:
            if spectra.dic.get("_peakfit_format") == "bruker":
                raise ValueError(
                    f"DQF-COSY: set fitting.dqfcosy_windows.{self.axis} for Bruker input. "
                    "Use retained complex points and the actual processing window."
                )
            window = _header_window(self._grid.spec_params)
        self.window: DQFCOSYWindow = window
        self.region_radius_hz = config.dqfcosy_region_hz

    def _parameter_id(self, label: str) -> ParameterId | None:
        if label == "j":
            return ParameterId(peak_name=self.name, axis=self._coupling_axis, label=label)
        return super()._parameter_id(label)

    def _param_name_for_spec(self, spec_name: str, shape: Any) -> str:
        if spec_name == "j":
            return ParameterId(peak_name=shape.name, axis=self._coupling_axis, label="j").name
        return super()._param_name_for_spec(spec_name, shape)

    def evaluate_offsets(
        self,
        offsets_hz: FloatArray,
        sign: FloatArray,
        cluster_params: ClusterParameters,
        compute_derivs: bool = False,
    ) -> LineshapeResult:
        """Evaluate the phased antiphase doublet on center-relative Hz offsets."""
        lw = cluster_params.extras["lw"]
        coupling = cluster_params.extras["j"]
        phases = cluster_params.extras.get("phase", np.zeros(cluster_params.n_peaks))
        rotation = np.exp(1j * np.deg2rad(phases))
        values, derivatives = antiphase_profile(offsets_hz, lw, coupling, self.window)
        phased = values * rotation
        if not compute_derivs:
            return LineshapeResult(values=sign * phased.real)
        return LineshapeResult(
            values=sign * phased.real,
            derivatives={
                "offset": sign * (rotation * derivatives["dw"]).real,
                "lw": sign * (rotation * derivatives["lw"]).real,
                "j": sign * (rotation * derivatives["j"]).real,
                "phase": sign * (1j * np.pi / 180 * phased).real,
            },
        )
