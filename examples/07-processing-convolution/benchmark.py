"""Benchmark sampled frequency-domain convolution against the direct window model.

This is an experimental comparison, not a replacement for routine fitting.
Frequency sampling approximates the continuous intrinsic profile; the direct
model remains the reference for assessing that error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.fft import fft, ifft, next_fast_len

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.fitting.optimizers import fit_cluster
from peakfit.fit.fitting import load_data

if TYPE_CHECKING:
    from peakfit.engine.domain.params_scalar import Parameters
    from peakfit.engine.lineshapes.base import ShapeBase


class ConvolutionCluster(Cluster):
    """Experimental FFT convolution of sampled intrinsic pseudo-Voigt peaks."""

    def prepare(self, oversample: int = 1) -> None:
        """Cache window kernels on the acquisition-period frequency lattice."""
        self.shapes = [cast("ShapeBase", shape) for shape in self.peaks[0].shapes]
        self.lengths = []
        self.filters = []
        self.frequencies = []
        self.origins = []
        self.steps = []
        for shape in self.shapes:
            spec = shape._grid.spec_params
            window = shape.processing_window
            assert window is not None
            # The spectral lattice must include the entire acquisition period.
            length = round(1 / window.dwell / abs(spec.sw / spec.size)) * oversample
            length = max(length, next_fast_len(2 * window.points - 1))
            self.lengths.append(length)
            period = 1 / window.dwell
            step = period / length
            self.steps.append(step)
            origin = shape.center * spec.obs + (spec.pts2hz(0.0) - shape.center * spec.obs) % step
            self.origins.append(origin)
            self.frequencies.append(origin + np.fft.fftfreq(length, window.dwell))
            weights = np.zeros(length)
            samples = window.samples()
            weights[: window.points] = samples
            weights[0] *= 2
            weights[-window.points + 1 :] = samples[:0:-1]
            # A frequency convolution kernel; FFT convolution uses its inverse
            # transform as a fixed multiplier. No processing-window resampling.
            kernel = fft(weights)
            self.filters.append(ifft(kernel).real)

    def _convolve(self, intrinsic: np.ndarray, grids: list[np.ndarray]) -> np.ndarray:
        # Convolve the first axis, then discard unrequested rows before doing
        # the second axis. This avoids a second transform of the entire plane.
        first = fft(ifft(intrinsic, axis=0) * self.filters[0][:, None], axis=0).real
        indices = []
        for shape, grid, length, step, origin in zip(
            self.shapes, grids, self.lengths, self.steps, self.origins, strict=True
        ):
            spec = shape._grid.spec_params
            coordinates = (spec.pts2hz(grid.astype(float)) - origin) / step
            if not np.allclose(coordinates, np.round(coordinates), atol=1e-7):
                raise ValueError("Output grid must lie on the convolution lattice.")
            indices.append(np.round(coordinates).astype(int) % length)
        unique, inverse = np.unique(indices[0], return_inverse=True)
        second = fft(ifft(first[unique], axis=1) * self.filters[1][None, :], axis=1).real
        return second[inverse, indices[1]]

    def _intrinsic(self, peak, params: Parameters, labels, start, stop) -> dict:
        obs = [shape._grid.spec_params.obs for shape in peak.shapes]
        centers = [
            params[shape.parameter_name("cs")].value * sf
            for shape, sf in zip(peak.shapes, obs, strict=True)
        ]
        widths = [params[shape.parameter_name("lw")].value for shape in peak.shapes]
        etas = [params[shape.parameter_name("eta")].value for shape in peak.shapes]
        angle = np.deg2rad(params[peak.tilt_parameter_id.name].value if peak.tilted else 0.0)
        c, s = np.cos(angle), np.sin(angle)
        x = self.frequencies[0][:, None] - centers[0]
        y = self.frequencies[1][None, start:stop] - centers[1]
        u, v = c * x + s * obs[0] / obs[1] * y, -s * obs[1] / obs[0] * x + c * y
        px, dx = _profile(u, widths[0], etas[0])
        py, dy = _profile(v, widths[1], etas[1])
        result = {}
        for label in labels:
            if label is None:
                value = px * py
            elif label == peak.tilt_parameter_id.name:
                value = (
                    (
                        dx["offset"] * (-s * x + c * obs[0] / obs[1] * y) * py
                        + px * dy["offset"] * (-c * obs[1] / obs[0] * x - s * y)
                    )
                    * np.pi
                    / 180
                )
            else:
                axis = 0 if label.startswith(f"{peak.name}.{peak.shapes[0].axis}.") else 1
                quantity = label.rsplit(".", 1)[-1]
                if quantity == "cs":
                    value = (
                        dx["offset"] * (-obs[0] * c) * py + px * dy["offset"] * (obs[1] * s)
                        if axis == 0
                        else dx["offset"] * (-obs[0] * s) * py + px * dy["offset"] * (-obs[1] * c)
                    )
                else:
                    value = dx[quantity] * py if axis == 0 else px * dy[quantity]
            result[label] = value
        return result

    def evaluate(
        self, params, grid_indices=None, d_matrix=None, param_map=None, component_indices=None
    ):
        """Convolve intrinsic values and analytic derivatives at requested grid points."""
        if component_indices is not None:
            raise ValueError("This experiment fits real spectra only.")
        grids = self.grid_indices if grid_indices is None else grid_indices
        indices = []
        for shape, grid, length, step, origin in zip(
            self.shapes, grids, self.lengths, self.steps, self.origins, strict=True
        ):
            coordinates = (shape._grid.spec_params.pts2hz(grid.astype(float)) - origin) / step
            if not np.allclose(coordinates, np.round(coordinates), atol=1e-7):
                raise ValueError("Output grid must lie on the convolution lattice.")
            indices.append(np.round(coordinates).astype(int) % length)
        unique, inverse = np.unique(indices[0], return_inverse=True)
        values = []
        for peak in self.peaks:
            labels = [None]
            if d_matrix is not None and param_map is not None:
                labels += [
                    name
                    for name in peak.create_params()
                    if param_map.get(name) is not None
                    and not name.rsplit(".", 1)[-1].startswith("I")
                ]
            intermediate = {label: np.empty((unique.size, self.lengths[1])) for label in labels}
            for start in range(0, self.lengths[1], 128):
                stop = min(start + 128, self.lengths[1])
                blocks = self._intrinsic(peak, params, labels, start, stop)
                for label, array in blocks.items():
                    transformed = fft(ifft(array, axis=0) * self.filters[0][:, None], axis=0).real
                    intermediate[label][:, start:stop] = transformed[unique]
            for label, array in intermediate.items():
                transformed = fft(ifft(array, axis=1) * self.filters[1][None, :], axis=1).real
                result = transformed[inverse, indices[1]]
                if label is None:
                    values.append(result)
                else:
                    assert d_matrix is not None
                    assert param_map is not None
                    d_matrix[param_map.get(label)] += result
        return np.asarray(values)


def _profile(offset, width, eta):
    gamma = width / 2
    ratio = offset / gamma
    gaussian = np.exp(-np.log(2) * ratio**2)
    lorentzian = 1 / (1 + ratio**2)
    return (1 - eta) * gaussian + eta * lorentzian, {
        "offset": (1 - eta) * gaussian * (-2 * np.log(2) * offset / gamma**2)
        - eta * lorentzian**2 * 2 * offset / gamma**2,
        "lw": (1 - eta) * gaussian * np.log(2) * offset**2 / gamma**3
        + eta * lorentzian**2 * offset**2 / gamma**3,
        "eta": lorentzian - gaussian,
    }


def compare(path: Path, peak_name: str, output: Path) -> None:
    """Compare same-mask evaluations, accuracy, and complete one-cluster fits."""
    config = PeakFitConfig(
        fitting=FitConfig(lineshape="pvoigt", include_window_functions=True, tilted=True)
    )
    folder = path.parent if path.is_file() else path
    loaded = load_data(path, folder / "peaklist.xml", None, config)
    original = next(c for c in loaded.clusters if [p.name for p in c.peaks] == [peak_name])
    params = original.peaks[0].create_params()
    params[f"{peak_name}.F2.lw"].value = 51
    params[f"{peak_name}.F3.lw"].value = 100
    params[original.peaks[0].tilt_parameter_id.name].value = 20
    names = params.get_vary_names()
    mapping = ParameterMap.from_names(names)
    jac = np.zeros((len(names), original.n_points))
    reference = original.evaluate(params, d_matrix=jac, param_map=mapping)
    rows = []
    for oversample in (1,):
        candidate = ConvolutionCluster(
            original.cluster_id, original.peaks, original.grid_indices, original.data
        )
        candidate.prepare(oversample)
        derivative = np.zeros_like(jac)
        started = time.perf_counter()
        values = candidate.evaluate(params, d_matrix=derivative, param_map=mapping)
        seconds = time.perf_counter() - started
        row = {
            "oversample": oversample,
            "lattice": candidate.lengths,
            "evaluation_seconds": seconds,
            "relative_value_error": float(np.max(abs(values - reference)) / np.max(abs(reference))),
            "relative_jacobian_errors": {
                name: float(np.max(abs(derivative[i] - jac[i])) / max(np.max(abs(jac[i])), 1e-10))
                for i, name in enumerate(names)
            },
        }
        sys.stdout.write(json.dumps(row) + "\n")
        sys.stdout.flush()
        rows.append(row)
        if oversample == 1:
            models = [("direct", original), ("convolution", candidate)]
            fit_rows = []
            for method, model in models:
                started = time.perf_counter()
                fitted = fit_cluster(params.copy(deep=True), model, noise=loaded.noise, max_nfev=50)
                fit_rows.append(
                    {
                        "method": method,
                        "seconds": time.perf_counter() - started,
                        "nfev": fitted.nfev,
                        "success": fitted.success,
                        "redchi": fitted.redchi,
                        "parameters": {name: fitted.params[name].value for name in names},
                    }
                )
                sys.stdout.write(json.dumps(fit_rows[-1]) + "\n")
                sys.stdout.flush()
            rows.append({"fits": fit_rows})
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path)
    parser.add_argument("--peak", default="peak2")
    parser.add_argument("--output", type=Path, default=Path("/tmp/processing-convolution.json"))
    args = parser.parse_args()
    compare(args.spectrum, args.peak, args.output)
