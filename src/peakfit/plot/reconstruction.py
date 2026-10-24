"""Full-grid spectra reconstructed from final fit values and saved geometry."""

import json
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.io.readers.results import ResultsLoader
from peakfit.io.spectrum_input import read_extracted_spectrum
from peakfit.io.state import default_state_path, load_state

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.domain.state import FittingState
    from peakfit.io.schemas import FitSummarySchema
    from peakfit.shared.typing import FloatArray

_BLOCK_POINTS = 262_144


class SpectraReconstructor:
    """Evaluate saved lineshapes with canonical final parameters and amplitudes."""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self._loader = ResultsLoader(results_dir)
        self._state: FittingState | None = None
        self._cluster_all: Cluster | None = None
        self._amplitudes: FloatArray | None = None
        self._parameters = Parameters()
        self._region_spectra: Spectra | None = None
        self._region_peaks: list[Peak] = []

    @property
    def has_region(self) -> bool:
        """Whether the export supplies a crop and processing geometry sidecar."""
        return (self.results_dir / "metadata" / "extraction.json").is_file()

    def load_region(self, source_path: Path) -> Spectra:
        """Load and verify the recorded crop from a supplied source spectrum."""
        self._region_spectra = read_extracted_spectrum(
            source_path, self.results_dir / "metadata" / "extraction.json"
        )
        self._amplitudes = None
        self._region_peaks = []
        return self._region_spectra

    @property
    def noise(self) -> float:
        """Use the completed summary's noise rather than estimating it from a crop."""
        return self._loader.load_summary().noise

    @property
    def peaks(self) -> list[Peak]:
        """Return verified model geometry, with positions from final parameters."""
        self._prepare_model()
        if self._cluster_all is not None:
            return self._cluster_all.peaks
        return self._region_peaks

    @property
    def _components(self) -> tuple[str, ...]:
        return self._cluster_all.quadrature_components if self._cluster_all is not None else ()

    @property
    def state(self) -> FittingState:
        """Load geometry from saved state; summary-only placeholder shapes cannot simulate."""
        if self._state is None:
            state_path = default_state_path(self.results_dir)
            if not state_path.exists():
                raise ValueError(
                    "Simulated contours require metadata/fitting_state.pkl. "
                    "Use a results directory containing saved fitting state, "
                    "or omit --results to view experimental data."
                )
            self._state = load_state(state_path)
        return self._state

    def _prepare_model(self) -> None:
        if self._amplitudes is not None:
            return
        summary = self._loader.load_summary()
        params = Parameters()
        for parameter in summary.final_nonlinear_parameters:
            params.add(parameter.name, value=parameter.value, unit=parameter.unit)
        self._parameters = params
        if self.has_region:
            self._prepare_region_model(summary, params)
            return
        by_id = {cluster.cluster_id: cluster for cluster in self.state.clusters}
        clusters = []
        amplitudes = []
        for outcome in summary.clusters:
            evaluation = outcome.analytical_evaluation
            if evaluation is None:
                continue
            cluster = by_id.get(outcome.cluster_id)
            if cluster is None or [peak.name for peak in cluster.peaks] != outcome.peak_names:
                raise ValueError("Saved geometry does not match the final summary's cluster peaks.")
            clusters.append(cluster)
            amplitudes.append(np.asarray(evaluation.amplitudes, dtype=np.float64))
        if not clusters:
            raise ValueError("Results contain no usable fitted peaks to simulate.")
        self._cluster_all = Cluster.from_clusters(clusters)
        self._amplitudes = np.concatenate(amplitudes)
        self.state.scalar_params = params
        self.state.noise = summary.noise

    def _prepare_region_model(self, summary: FitSummarySchema, params: Parameters) -> None:
        spectra = self._region_spectra
        if spectra is None:
            raise ValueError(
                "Load the saved region from its source before reconstructing contours."
            )
        metadata = json.loads((self.results_dir / "metadata" / "extraction.json").read_text())
        models = metadata["models"]
        expected = [param.label for param in spectra.spectral_params]
        if [model["internal_label"] for model in models] != expected:
            raise ValueError("Saved model axes do not match the extracted frequency axes.")
        names = [model["selected"] for model in models]
        config = PeakFitConfig.model_validate(deepcopy(summary.metadata.configuration))
        peaks = []
        amplitudes = []
        for outcome in summary.clusters:
            evaluation = outcome.analytical_evaluation
            if evaluation is None:
                continue
            for name in outcome.peak_names:
                positions = tuple(params[f"{name}.{axis}.cs"].value for axis in expected)
                peak = Peak(
                    name=name,
                    positions=np.array(positions),
                    shapes=create_shapes(spectra, config.fitting, name, positions, names),
                    tilted=config.fitting.tilted,
                )
                peak.set_cluster_id(outcome.cluster_id)
                peaks.append(peak)
            amplitudes.append(np.asarray(evaluation.amplitudes, dtype=np.float64))
        if not peaks:
            raise ValueError("Results contain no usable fitted peaks to simulate.")
        self._region_peaks = peaks
        self._amplitudes = np.concatenate(amplitudes)

    def validate_spectrum(
        self,
        shape: tuple[int, ...],
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        """Reject mismatched plane counts, sampling grids, and chemical-shift axes."""
        self._prepare_model()
        assert self._amplitudes is not None
        if self._components:
            if shape[0] != 1:
                raise ValueError("Quadrature results currently support viewing the real rr plane.")
        elif self._amplitudes.shape[1] != shape[0]:
            raise ValueError("Fit results and experimental spectrum have different plane counts.")
        for peak in self.peaks:
            if len(peak.shapes) != len(shape) - 1:
                raise ValueError("Fit results and spectrum have different spectral dimensions.")
            for axis, size, limits in zip(peak.shapes, shape[1:], (ylim, xlim), strict=True):
                context = axis.dim_ctx
                if context.size != size or not np.allclose(
                    context.ppm2pts(np.asarray(limits)), [0, size - 1], rtol=0, atol=1e-4
                ):
                    raise ValueError(
                        "Fit results and experimental spectrum have different ppm grids."
                    )

    def reconstruct_plane(self, plane_index: int, grid_shape_2d: tuple[int, int]) -> FloatArray:
        """Evaluate one full-resolution plane, retaining tails and signed intensities.

        Point blocks bound temporary memory. No linewidth cutoff is applied,
        because clipped tails would create artificial residual contours.
        """
        self._prepare_model()
        assert self._amplitudes is not None
        components = self._components
        count = len(components) if components else self._amplitudes.shape[1]
        if not 0 <= plane_index < count:
            raise ValueError("Plane index is outside the fitted model.")
        amplitudes = self._amplitudes[:, 0 if components else plane_index]
        simulated = np.zeros(grid_shape_2d, dtype=np.float64)
        output = simulated.ravel()
        for start in range(0, output.size, _BLOCK_POINTS):
            stop = min(start + _BLOCK_POINTS, output.size)
            points = np.arange(start, stop, dtype=np.intp)
            positions = [points // grid_shape_2d[1], points % grid_shape_2d[1]]
            component_indices = (
                np.full(points.size, plane_index, dtype=np.intp) if components else None
            )
            # Evaluate peaks separately to bound memory even for large peak lists.
            for peak, amplitude in zip(self.peaks, amplitudes, strict=True):
                if amplitude == 0:
                    continue
                values = Cluster.evaluate_peaks(
                    [peak],
                    self._parameters,
                    positions,
                    quadrature_components=components,
                    component_indices=component_indices,
                )[0]
                output[start:stop] += amplitude * values
        return simulated
