"""Sampled forward transforms of intrinsic peaks on the acquisition axes.

The two-sided transform of an absorptive intrinsic profile supplies its time
signal. A symmetric extension of each recorded one-sided window supplies the
real absorption spectrum. Rotating the intrinsic profile rotates its dual time
coordinates, while the processing windows remain on the acquisition axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.param_map import ParameterMap
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.domain.peaks import Peak
    from peakfit.engine.domain.window import ProcessingWindow
    from peakfit.engine.lineshapes.base import ShapeBase
    from peakfit.shared.typing import FloatArray, IntArray


def _time_axis(window: ProcessingWindow) -> tuple[FloatArray, FloatArray]:
    t = np.arange(1 - window.points, window.points, dtype=float) * window.dwell
    samples = window.samples()
    weights = np.concatenate((samples[:0:-1], samples))
    # The zero sample replaces the two half-axis contributions, each scaled by C1.
    weights[window.points - 1] *= 2
    return t, weights * window.dwell


def _intrinsic_signal(
    t: FloatArray, width: float, kind: str, eta: float, coupling: float
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Fourier transform of unit-height profiles, with width, time, and mix derivatives."""
    b = np.pi**2 / (4 * np.log(2))
    gaussian = width * (np.pi / (4 * np.log(2))) ** 0.5 * np.exp(-b * (width * t) ** 2)
    lorentzian = np.pi * width / 2 * np.exp(-np.pi * width * np.abs(t))
    if kind.startswith("gaussian"):
        eta = 0.0
    elif kind.startswith("lorentzian"):
        eta = 1.0
    value = (1 - eta) * gaussian + eta * lorentzian
    derivatives = {
        "lw": (1 - eta) * gaussian * (1 / width - 2 * b * width * t**2)
        + eta * lorentzian * (1 / width - np.pi * np.abs(t)),
        "time": (1 - eta) * gaussian * (-2 * b * width**2 * t)
        - eta * lorentzian * np.pi * width * np.sign(t),
    }
    if kind.startswith("pvoigt"):
        derivatives["eta"] = lorentzian - gaussian
    if kind.endswith("doublet"):
        modulation = 2 * np.cos(np.pi * coupling * t)
        derivatives["time"] = derivatives["time"] * modulation - value * (
            2 * np.pi * coupling * np.sin(np.pi * coupling * t)
        )
        derivatives["j"] = -2 * value * np.pi * t * np.sin(np.pi * coupling * t)
        for label in ("lw", "eta"):
            if label in derivatives:
                derivatives[label] *= modulation
        value *= modulation
    return value, derivatives


def _transform(signal: np.ndarray, bases: list[np.ndarray], inverses: list[IntArray]) -> FloatArray:
    if len(bases) == 1:
        return (bases[0] @ signal).real[inverses[0]]
    # Choose the multiplication order with the smaller intermediate array.
    if bases[0].shape[0] * signal.shape[1] <= bases[1].shape[0] * signal.shape[0]:
        transformed = (bases[0] @ signal) @ bases[1].T
    else:
        transformed = bases[0] @ (signal @ bases[1].T)
    return transformed.real[inverses[0], inverses[1]]


def evaluate_processed_peaks(
    peaks: Sequence[Peak],
    params: Parameters,
    grids: Sequence[IntArray],
    d_matrix: FloatArray | None,
    param_map: ParameterMap | None,
) -> FloatArray:
    """Evaluate sampled peaks and derivatives at available frequency coordinates."""
    shape_axes = [cast("ShapeBase", shape) for shape in peaks[0].shapes]
    times = []
    weights = []
    bases = []
    inverses = []
    for shape, grid in zip(shape_axes, grids, strict=True):
        assert shape.processing_window is not None
        t, weight = _time_axis(shape.processing_window)
        unique, inverse = np.unique(grid, return_inverse=True)
        frequencies = shape._grid.spec_params.pts2hz(unique.astype(float))
        # Subtract a fixed reference to keep Fourier phases small.
        reference = shape.center * shape._grid.spec_params.obs
        bases.append(np.exp(-2j * np.pi * (frequencies - reference)[:, None] * t))
        inverses.append(inverse)
        times.append(t)
        weights.append(weight)
    fixed_weights = weights[0][:, None] * weights[1][None, :] if len(grids) > 1 else weights[0]

    values = []
    for peak in peaks:
        if not peak.tilted:
            value, derivatives = _axis_aligned_values(
                peak, params, times, weights, bases, inverses, shape_axes
            )
            values.append(value)
            if d_matrix is not None and param_map is not None:
                for name, derivative in derivatives.items():
                    index = param_map.get(name)
                    if index is not None:
                        d_matrix[index] += derivative
            continue
        signal, derivatives = _peak_signal(
            peak,
            params,
            times,
            shape_axes,
            fixed_weights,
            compute_derivatives=d_matrix is not None and param_map is not None,
        )
        values.append(_transform(signal, bases, inverses))
        if d_matrix is not None and param_map is not None:
            for name, derivative in derivatives.items():
                index = param_map.get(name)
                if index is not None:
                    d_matrix[index] += _transform(derivative, bases, inverses)
    return np.asarray(values)


def _peak_signal(
    peak: Peak,
    params: Parameters,
    times: list[FloatArray],
    shape_axes: list[ShapeBase],
    fixed_weights: FloatArray,
    *,
    compute_derivatives: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    shapes = [cast("ShapeBase", shape) for shape in peak.shapes]
    widths = [params[shape.parameter_name("lw")].value for shape in shapes]
    angle = params[peak.tilt_parameter_id.name].value if peak.tilted else 0.0
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    tx = times[0] if len(times) == 1 else times[0][:, None]
    if len(times) > 1:
        ty = times[1][None, :]
        obs_x, obs_y = [shape._grid.spec_params.obs for shape in shapes]
        # Dual of u=obs_x*(c*x/obs_x+s*y/obs_y), v=obs_y*(-s*x/obs_x+c*y/obs_y).
        dual_times = [c * tx + s * obs_y / obs_x * ty, -s * obs_x / obs_y * tx + c * ty]
        dual_derivatives = [-s * tx + c * obs_y / obs_x * ty, -c * obs_x / obs_y * tx - s * ty]
    else:
        dual_times = [tx]
        dual_derivatives = []
    signals = []
    signal_derivatives = []
    translation = np.ones_like(fixed_weights, dtype=complex)
    for axis, (shape, width, t) in enumerate(zip(shapes, widths, dual_times, strict=True)):
        kind = shape.shape_name
        eta = params[shape.parameter_name("eta")].value if kind.startswith("pvoigt") else 0
        j = params[shape.parameter_name("j")].value if kind.endswith("doublet") else 0
        signal, derivatives = _intrinsic_signal(t, width, kind, eta, j)
        signals.append(signal)
        signal_derivatives.append(derivatives)
        shift = (params[shape.parameter_name("cs")].value - shape_axes[axis].center) * (
            shape._grid.spec_params.obs
        )
        axis_time = (
            times[axis]
            if len(times) == 1
            else (times[axis][:, None] if axis == 0 else times[axis][None, :])
        )
        translation *= np.exp(2j * np.pi * shift * axis_time)
    phase_factors, phase_derivatives = _phase_factors(shapes, params, times)
    phase_product = phase_factors[0] if len(times) == 1 else phase_factors[0] * phase_factors[1]
    factor = fixed_weights * translation * phase_product
    product = signals[0] if len(times) == 1 else signals[0] * signals[1]
    if not compute_derivatives:
        return factor * product, {}
    result_derivatives = {}
    for axis, shape in enumerate(shapes):
        other = signals[1 - axis] if len(times) > 1 else 1.0
        name = shape.parameter_name("phase")
        if name in params:
            other_phase = phase_factors[1 - axis] if len(times) > 1 else 1.0
            result_derivatives[name] = (
                fixed_weights * translation * product * phase_derivatives[axis] * other_phase
            )
        for label, derivative in signal_derivatives[axis].items():
            if label != "time":
                result_derivatives[shape.parameter_name(label)] = factor * derivative * other
        axis_time = (
            times[axis]
            if len(times) == 1
            else (times[axis][:, None] if axis == 0 else times[axis][None, :])
        )
        result_derivatives[shape.parameter_name("cs")] = (
            factor * product * 2j * np.pi * shape._grid.spec_params.obs * axis_time
        )
    if peak.tilted:
        derivative = (
            (
                signal_derivatives[0]["time"] * dual_derivatives[0] * signals[1]
                + signals[0] * signal_derivatives[1]["time"] * dual_derivatives[1]
            )
            * np.pi
            / 180
        )
        result_derivatives[peak.tilt_parameter_id.name] = factor * derivative
    return factor * product, result_derivatives


def _axis_time(times: list[FloatArray], axis: int) -> FloatArray:
    if len(times) == 1:
        return times[0]
    return times[axis][:, None] if axis == 0 else times[axis][None, :]


def _phase_factors(
    shapes: list[ShapeBase],
    params: Parameters,
    times: list[FloatArray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    phase_factors = []
    phase_derivatives = []
    for axis, shape in enumerate(shapes):
        axis_time = _axis_time(times, axis)
        name = shape.parameter_name("phase")
        phase = np.deg2rad(params[name].value) if name in params else 0.0
        sign = np.sign(axis_time)
        multiplier = np.exp(1j * phase * sign)
        derivative = 1j * sign * multiplier * np.pi / 180
        multiplier = np.where(sign == 0, np.cos(phase), multiplier)
        derivative = np.where(sign == 0, -np.sin(phase) * np.pi / 180, derivative)
        phase_factors.append(multiplier)
        phase_derivatives.append(derivative)
    return phase_factors, phase_derivatives


def _axis_aligned_values(
    peak: Peak,
    params: Parameters,
    times: list[FloatArray],
    weights: list[FloatArray],
    bases: list[np.ndarray],
    inverses: list[IntArray],
    references: list[ShapeBase],
) -> tuple[FloatArray, dict[str, FloatArray]]:
    profiles = []
    derivatives = []
    for axis, shape in enumerate(peak.shapes):
        value, derivative = _axis_profile(
            cast("ShapeBase", shape),
            params,
            times[axis],
            weights[axis],
            bases[axis],
            inverses[axis],
            references[axis].center,
        )
        profiles.append(value)
        derivatives.append(derivative)
    value = profiles[0] if len(profiles) == 1 else profiles[0] * profiles[1]
    result_derivatives = {}
    for axis, derivative in enumerate(derivatives):
        other = profiles[1 - axis] if len(profiles) > 1 else 1.0
        for name, values in derivative.items():
            result_derivatives[name] = values * other
    return value, result_derivatives


def _axis_profile(
    shape: ShapeBase,
    params: Parameters,
    t: FloatArray,
    weights: FloatArray,
    basis: np.ndarray,
    inverse: IntArray,
    reference: float,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    kind = shape.shape_name
    width = params[shape.parameter_name("lw")].value
    eta = params[shape.parameter_name("eta")].value if kind.startswith("pvoigt") else 0
    j = params[shape.parameter_name("j")].value if kind.endswith("doublet") else 0
    signal, derivatives = _intrinsic_signal(t, width, kind, eta, j)
    phase_factors, phase_derivatives = _phase_factors([shape], params, [t])
    shift = (params[shape.parameter_name("cs")].value - reference) * shape._grid.spec_params.obs
    factor = weights * np.exp(2j * np.pi * shift * t)
    coefficients = signal * factor * phase_factors[0]
    value = (basis @ coefficients).real[inverse]
    result_derivatives = {
        shape.parameter_name(label): (basis @ (factor * phase_factors[0] * derivative)).real[
            inverse
        ]
        for label, derivative in derivatives.items()
        if label != "time"
    }
    result_derivatives[shape.parameter_name("cs")] = (
        basis @ (coefficients * 2j * np.pi * shape._grid.spec_params.obs * t)
    ).real[inverse]
    name = shape.parameter_name("phase")
    if name in params:
        result_derivatives[name] = (basis @ (signal * factor * phase_derivatives[0])).real[inverse]
    return value, result_derivatives
