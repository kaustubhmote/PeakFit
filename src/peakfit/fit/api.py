"""Headless preparation and fitting through the existing PeakFit pipeline."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import numpy as np

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.param_id import ParameterId
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import get_dimension_label
from peakfit.engine.lineshapes.create import SHAPES, create_shapes
from peakfit.fit.fitting import fit_data, prepare_data
from peakfit.fit.output_metadata import RunMetadata
from peakfit.fit.run_models import LoadedData, RunSummary
from peakfit.fit.simulation import FinalModelSnapshot, simulate_final_outcome
from peakfit.io.readers.peaks import read_list
from peakfit.io.spectrum_input import (
    PPM_TOLERANCE,
    Extraction,
    Region,
    SpectrumInput,
    extract_spectrum,
    plan_extraction,
)
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import write_fit_outputs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peakfit.fit.final_outcome import FinalFitOutcome, FinalParameter
    from peakfit.io.spectrum_input import AxisProcessing, SeriesAxis

_TWO_FREQUENCY_AXES = 2

_EMPIRICAL = (
    "gaussian",
    "lorentzian",
    "pvoigt",
    "gaussian_doublet",
    "lorentzian_doublet",
    "pvoigt_doublet",
)
_PROCESSING_FIELDS = (
    "acquisition_time",
    "td_size",
    "ft_size",
    "apod_code",
    "phase0_deg",
    "phase1_deg",
)


class FitCancelledError(RuntimeError):
    """A caller cancelled preparation or fitting before a completed result existed."""


@dataclass(frozen=True)
class PeakSeed:
    """Stable ID and ppm positions keyed by original ndarray axes.

    widths_hz optionally supplies starting FWHM values and finite neighbor-support
    diagnostics. sign is a provenance hint, not an amplitude constraint.
    """

    id: str
    positions_ppm: Mapping[int, float]
    widths_hz: Mapping[int, float] = field(default_factory=dict)
    sign: int | None = None
    label: str | None = None
    provenance: str = "caller"

    def __post_init__(self) -> None:
        """Copy records and validate identifiers and numeric values."""
        if not self.id or "." in self.id or self.id.startswith("cluster_"):
            raise ValueError(
                "Seed IDs must be nonempty, contain no '.', and not start with 'cluster_'."
            )
        positions = dict(self.positions_ppm)
        widths = dict(self.widths_hz)
        if (
            not positions
            or any(type(key) is not int for key in positions)
            or not np.isfinite(list(positions.values())).all()
        ):
            raise ValueError("Seed positions require external integer axes and finite ppm values.")
        if set(widths) - set(positions) or any(
            not np.isfinite(v) or v <= 0 for v in widths.values()
        ):
            raise ValueError("Seed widths must be positive finite FWHM values in Hz on known axes.")
        if self.sign not in {None, -1, 1}:
            raise ValueError("Seed sign must be -1 or +1.")
        object.__setattr__(self, "positions_ppm", MappingProxyType(positions))
        object.__setattr__(self, "widths_hz", MappingProxyType(widths))


@dataclass(frozen=True)
class SeedDecision:
    """Explicit seed inclusion and analysis-region membership."""

    seed: PeakSeed
    included: bool
    in_analysis_region: bool
    reason: str
    boundary_axes: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModelAvailability:
    """Per-external-axis model selection, available models, and rejection reasons."""

    external_axis: int
    internal_label: str
    selected: str
    available: tuple[str, ...]
    unavailable: Mapping[str, str]
    missing_fields: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class PreparedFit:
    """Prepared crop, validated configuration, seeds, and clusters without fitting."""

    data: LoadedData
    extraction: Extraction
    seeds: tuple[SeedDecision, ...]
    models: tuple[ModelAvailability, ...]
    config: PeakFitConfig = field(repr=False)
    diagnostics: tuple[str, ...] = ()


def _model_rejection(model, proc, missing, fitting, label, dimensions) -> str:
    reason = ""
    if model == "dqfcosy":
        window = fitting.dqfcosy_windows.get(label) or (proc.dqfcosy_window if proc else None)
        if dimensions != _TWO_FREQUENCY_AXES or fitting.tilted:
            reason = "DQF-COSY requires exactly two frequency axes and tilted=false."
        elif window is None:
            reason = "Missing validated DQF-COSY acquisition/apodization window."
    elif missing:
        reason = "Missing processing metadata: " + ", ".join(missing)
    elif model.startswith("no_apod") and proc.apod_code != 0:
        reason = "no_apod requires a documented unapodized acquisition."
    elif model.startswith(("sp1", "sp2")):
        power = 1 if model.startswith("sp1") else 2
        if (
            proc.apod_code != 1
            or proc.apod_q3 != power
            or proc.apod_q1 is None
            or proc.apod_q2 is None
        ):
            reason = f"Requires complete sine-window coefficients with power {power}."
        elif not 0 <= proc.apod_q1 < proc.apod_q2 <= 1:
            reason = "Sine-window coefficients require 0 <= q1 < q2 <= 1."
    return reason


def _processing_candidate(proc) -> str:
    candidate = "pvoigt"
    if proc is not None and proc.apod_code == 0:
        candidate = "no_apod"
    elif proc is not None and proc.apod_code == 1:
        if proc.apod_q3 == 1:
            candidate = "sp1"
        elif proc.apod_q3 == _TWO_FREQUENCY_AXES:
            candidate = "sp2"
    return candidate


def _automatic_model(proc, unavailable, missing, windowed) -> tuple[str, str]:
    if windowed:
        return "pvoigt", "Selected intrinsic pvoigt with the validated acquisition window."
    candidate = _processing_candidate(proc)
    selected = "pvoigt" if candidate in unavailable else candidate
    reason = unavailable.get(candidate, "Selected from validated processing metadata.")
    if candidate == "pvoigt":
        reason = "No supported acquisition-window model; selected empirical pvoigt."
        if missing:
            reason += " Missing: " + ", ".join(missing)
    return selected, reason


def model_availability(
    source: SpectrumInput,
    config: PeakFitConfig | None = None,
) -> tuple[ModelAvailability, ...]:
    """Inspect model requirements and selection without data allocation or fitting."""
    config = config or PeakFitConfig()
    windowed = config.fitting.include_window_functions
    if windowed and len(source.frequency_axes) > _TWO_FREQUENCY_AXES:
        raise ValueError("Window-aware intrinsic fitting supports one or two frequency axes.")
    reports = []
    for index, external_axis in enumerate(source.frequency_axes):
        label = get_dimension_label(index)
        proc = source.processing.get(external_axis)
        missing = tuple(
            key for key in _PROCESSING_FIELDS if proc is None or getattr(proc, key) is None
        )
        unavailable = {}
        for model in SHAPES:
            if model in _EMPIRICAL:
                continue
            reason = _model_rejection(
                model, proc, missing, config.fitting, label, len(source.frequency_axes)
            )
            if reason:
                unavailable[model] = reason
        requested = config.fitting.lineshape
        if windowed and requested != "dqfcosy":
            if proc is None or proc.processing_window is None:
                raise ValueError(
                    f"Axis {external_axis}: missing validated acquisition/window metadata "
                    "for include_window_functions."
                )
            for model in SHAPES:
                if model not in _EMPIRICAL:
                    unavailable[model] = "Window-aware intrinsic fitting requires empirical shapes."
        if requested == "auto":
            selected, reason = _automatic_model(proc, unavailable, missing, windowed)
        else:
            selected = requested
            if requested in unavailable:
                raise ValueError(
                    f"Axis {external_axis}: {requested} unavailable. {unavailable[requested]}"
                )
            reason = "Explicit caller selection."
        reports.append(
            ModelAvailability(
                external_axis,
                label,
                selected,
                tuple(model for model in SHAPES if model not in unavailable),
                MappingProxyType(unavailable),
                missing,
                reason,
            )
        )
    return tuple(reports)


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise FitCancelledError(
            "Fit cancelled by caller; no completed result or outputs were produced."
        )


def _validate_request(source, cfg, peaks, peaklist_path, auto_pick) -> None:
    if cfg.fitting.use_imaginary or cfg.fitting.imaginary_paths:
        raise ValueError(
            "This interface accepts real data. Use the DQF-COSY CLI for quadrature layouts."
        )
    if cfg.fitting.tilted and len(source.frequency_axes) != _TWO_FREQUENCY_AXES:
        raise ValueError("Tilted fitting requires exactly two frequency dimensions.")
    if peaks is not None and peaklist_path is not None:
        raise ValueError("Supply peaks or peaklist_path, not both.")
    if peaks is None and peaklist_path is None and not auto_pick:
        raise ValueError(
            "Supply peak seeds, peaklist_path, or explicitly enable experimental auto_pick."
        )


def _neighbor_centers(source, plan, seeds) -> tuple[Mapping[int, float], ...]:
    neighbors = []
    for seed in seeds:
        # A finite support rule must be explicit: four supplied FWHM per outside axis.
        overlaps = True
        outside = False
        for axis in plan.axes:
            position = seed.positions_ppm[axis.external_axis]
            low, high = sorted(axis.fitting_ppm)
            meta = source.udic[axis.external_axis]
            first = meta["car"] / meta["obs"] + meta["sw"] / (2 * meta["obs"])
            last = first - (meta["size"] - 1) * meta["sw"] / (meta["size"] * meta["obs"])
            if not last - PPM_TOLERANCE <= position <= first + PPM_TOLERANCE:
                overlaps = False
            if not low <= position <= high:
                outside = True
                margin = 4 * seed.widths_hz.get(axis.external_axis, 0) / meta["obs"]
                if position < low - margin or position > high + margin:
                    overlaps = False
        if overlaps and outside:
            neighbors.append(seed.positions_ppm)
    return tuple(neighbors)


def _select_noise(source, cfg, noise, noise_region_ppm, max_memory_bytes) -> str:
    if noise is not None and noise_region_ppm is not None:
        raise ValueError("Supply explicit noise or a noise region, not both.")
    noise_source = (
        "user-provided"
        if noise is not None or cfg.noise_level is not None
        else "estimated from fitting crop"
    )
    if noise is not None:
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("noise must be positive and finite.")
        cfg.noise_level = noise
    if noise_region_ppm is not None:
        if cfg.noise_level is not None:
            raise ValueError("noise_region_ppm conflicts with explicit configuration noise.")
        noise_data = extract_spectrum(source, noise_region_ppm, max_memory_bytes=max_memory_bytes)
        cfg.noise_level = float(estimate_noise(noise_data.spectra.data))
        noise_source = "estimated from separate source noise region"
    return noise_source


def _seed_decision(seed: PeakSeed, extraction: Extraction) -> SeedDecision:
    inside = all(
        min(a.fitting_ppm) - PPM_TOLERANCE
        <= seed.positions_ppm[a.external_axis]
        <= max(a.fitting_ppm) + PPM_TOLERANCE
        for a in extraction.axes
    )
    analysis = all(
        a.requested_ppm[0] - PPM_TOLERANCE
        <= seed.positions_ppm[a.external_axis]
        <= a.requested_ppm[1] + PPM_TOLERANCE
        for a in extraction.axes
    )
    boundary = tuple(
        a.external_axis
        for a in extraction.axes
        if min(abs(seed.positions_ppm[a.external_axis] - bound) for bound in a.fitting_ppm)
        <= PPM_TOLERANCE
    )
    reason = (
        "inside analysis region"
        if analysis
        else "neighbor inside fitting region, including halo or four supplied FWHM support"
    )
    if not inside:
        reason = "center outside fitting region; increase halo_points to include its tails"
    return SeedDecision(seed, inside, analysis, reason, boundary)


def _prepare_seeds(source, extracted, cfg, seeds, peaklist_path, names, supplied):
    spectra = extracted.spectra
    decisions = []
    domain_peaks = []
    if peaklist_path is not None:
        domain_peaks = read_list(Path(peaklist_path), spectra, names, cfg.fitting)
        seeds = tuple(
            PeakSeed(
                peak.name,
                {
                    axis: float(value)
                    for axis, value in zip(source.frequency_axes, peak.positions, strict=True)
                },
                provenance="peak list",
            )
            for peak in domain_peaks
        )
    if len({seed.id for seed in seeds}) != len(seeds):
        raise ValueError("Peak seed IDs must be unique, including peak-list records.")
    for seed in seeds:
        decision = _seed_decision(seed, extracted.extraction)
        decisions.append(decision)
        inside = decision.included
        if inside and peaklist_path is None:
            positions = tuple(seed.positions_ppm[axis] for axis in source.frequency_axes)
            domain_peaks.append(
                Peak(
                    name=seed.id,
                    positions=np.array(positions),
                    shapes=create_shapes(spectra, cfg.fitting, seed.id, positions, names),
                    tilted=cfg.fitting.tilted,
                )
            )
    if peaklist_path is not None:
        included = {d.seed.id for d in decisions if d.included}
        domain_peaks = [peak for peak in domain_peaks if peak.name in included]
    if (seeds and not domain_peaks) or (supplied and not seeds):
        raise ValueError("No usable peak seeds remain inside the fitting region.")
    return seeds, decisions, domain_peaks


def _estimate_peak_budget(plan, seeds, auto_pick, peaklist_path, cfg) -> int:
    peak_budget = max(1, len(seeds))
    if auto_pick and not seeds and peaklist_path is None:
        peak_budget = plan.resources.selected_points
        if cfg.auto_peak.max_peaks_per_roi is not None:
            peak_budget = min(
                peak_budget, cfg.auto_peak.max_clusters * cfg.auto_peak.max_peaks_per_roi
            )
    return peak_budget


def prepare_spectrum(
    source: SpectrumInput,
    *,
    peaks: Sequence[PeakSeed] | None = None,
    peaklist_path: str | Path | None = None,
    region_ppm: Region | None = None,
    config: PeakFitConfig | None = None,
    auto_pick: bool = False,
    halo_points: int = 0,
    noise: float | None = None,
    noise_region_ppm: Region | None = None,
    workers: int = 1,
    max_memory_bytes: int | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[Any], None] | None = None,
) -> PreparedFit:
    """Validate, crop, select models/seeds, estimate noise, and prepare clusters.

    Picking is experimental and opt-in. Noise uses the selected fit crop unless
    noise, config.noise_level, or a separate all-axis source noise region is supplied.
    """
    _check_cancel(cancel)
    cfg = PeakFitConfig.model_validate(config.model_dump()) if config else PeakFitConfig()
    _validate_request(source, cfg, peaks, peaklist_path, auto_pick)
    models = model_availability(source, cfg)
    if cfg.fitting.lineshape == "dqfcosy":
        for model in models:
            proc = source.processing.get(model.external_axis)
            if proc is not None and proc.dqfcosy_window is not None:
                cfg.fitting.dqfcosy_windows.setdefault(model.internal_label, proc.dqfcosy_window)
    seeds = tuple(peaks) if peaks is not None else ()
    if len({seed.id for seed in seeds}) != len(seeds):
        raise ValueError("Peak seed IDs must be unique.")
    for seed in seeds:
        if set(seed.positions_ppm) != set(source.frequency_axes):
            raise ValueError(
                f"Seed {seed.id} must provide positions on every external frequency axis."
            )
    plan = plan_extraction(
        source,
        region_ppm,
        halo_points=halo_points,
        workers=workers,
        peak_count=max(1, len(seeds)),
        exclude_planes=cfg.exclude_planes,
        max_memory_bytes=max_memory_bytes,
    )
    peak_budget = _estimate_peak_budget(plan, seeds, auto_pick, peaklist_path, cfg)
    neighbors = _neighbor_centers(source, plan, seeds)
    if neighbors or peak_budget != max(1, len(seeds)):
        plan = plan_extraction(
            source,
            region_ppm,
            halo_points=halo_points,
            workers=workers,
            peak_count=peak_budget,
            exclude_planes=cfg.exclude_planes,
            max_memory_bytes=max_memory_bytes,
            _include_centers=tuple(neighbors),
        )
    if progress is not None:
        progress(plan.resources)
    _check_cancel(cancel)
    extracted = extract_spectrum(
        source,
        region_ppm,
        halo_points=halo_points,
        workers=workers,
        peak_count=peak_budget,
        exclude_planes=cfg.exclude_planes,
        max_memory_bytes=max_memory_bytes,
        _include_centers=tuple(neighbors),
    )
    names = [model.selected for model in models]
    spectra = extracted.spectra
    seeds, decisions, domain_peaks = _prepare_seeds(
        source, extracted, cfg, seeds, peaklist_path, names, peaks is not None
    )
    noise_source = _select_noise(source, cfg, noise, noise_region_ppm, max_memory_bytes)
    for seed in seeds:
        for axis, width in seed.widths_hz.items():
            label = next(a.label for a in extracted.extraction.axes if a.external_axis == axis)
            # Seed starts use the existing parameter-constraint system.
            from peakfit.engine.domain.constraints import ParameterConstraint  # noqa: PLC0415

            cfg.parameters.defaults.patterns[f"{seed.id}.{label}.lw"] = ParameterConstraint(
                value=width
            )
    cfg.auto_peak.enabled = auto_pick
    _check_cancel(cancel)
    loaded = prepare_data(
        spectra,
        cfg,
        peaks=domain_peaks if seeds or peaks is not None else None,
        shape_names=names,
        cancel_check=lambda: _check_cancel(cancel),
    )
    if not np.isfinite(loaded.noise) or loaded.noise <= 0:
        raise ValueError(
            "Noise estimation failed; supply a positive noise value or a source noise region."
        )
    loaded = LoadedData(
        loaded.spectra,
        loaded.peaks,
        loaded.noise,
        noise_source,
        loaded.shape_names,
        loaded.contour_level,
        loaded.clusters,
    )
    if not decisions:
        decisions = [
            _seed_decision(
                PeakSeed(
                    peak.name,
                    {
                        axis: float(p)
                        for axis, p in zip(source.frequency_axes, peak.positions, strict=True)
                    },
                    provenance="experimental auto picker",
                ),
                extracted.extraction,
            )
            for peak in loaded.peaks
        ]
    _check_cancel(cancel)
    return PreparedFit(
        loaded,
        extracted.extraction,
        tuple(decisions),
        models,
        cfg,
        (*extracted.extraction.diagnostics, f"Noise: {noise_source}."),
    )


@dataclass(frozen=True)
class PeakEstimate:
    """Original-axis positions and signed amplitude rows projected from final truth."""

    id: str
    positions_ppm: Mapping[int, float]
    parameters: tuple[FinalParameter, ...]
    amplitudes: tuple[float, ...] | None
    amplitude_standard_errors: tuple[float, ...] | None
    classification: str
    seed: PeakSeed
    in_analysis_region: bool


@dataclass(frozen=True)
class SpectrumFitResult:
    """Completed immutable scientific outcome, original-axis identities, and provenance."""

    outcome: FinalFitOutcome
    peaks: tuple[PeakEstimate, ...]
    extraction: Extraction
    seeds: tuple[SeedDecision, ...]
    models: tuple[ModelAvailability, ...]
    diagnostics: tuple[str, ...]
    noise_source: str
    _snapshot: FinalModelSnapshot = field(repr=False)
    _data: np.ndarray = field(repr=False)
    _configuration: str = field(repr=False)

    def simulate(self) -> np.ndarray | None:
        """Evaluate only the cropped grid, in internal series/frequency order."""
        return simulate_final_outcome(self.outcome, self._snapshot, self._data)

    def residual(self) -> np.ndarray | None:
        """Return experimental minus model on the cropped grid, in intensity units."""
        simulated = self.simulate()
        return None if simulated is None else self._data - simulated

    def export(self, output_dir: str | Path) -> Mapping[str, Path]:
        """Write canonical final-outcome JSON/CSV, without continuation pickle files."""
        config = PeakFitConfig.model_validate_json(self._configuration)
        metadata = RunMetadata(configuration=config.model_dump(mode="json"))
        series = self.extraction.series
        values = np.array(series.values if series else [0.0])
        files = write_fit_outputs(
            self.outcome,
            Path(output_dir),
            WriterConfig(formats=tuple(config.output.formats)),
            metadata=metadata,
            z_values=values,
            summary=RunSummary.from_outcome(self.outcome),
        )
        provenance_path = Path(output_dir) / "metadata" / "extraction.json"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(
                _json_metadata(
                    {
                        "extraction": self.extraction,
                        "seeds": self.seeds,
                        "models": self.models,
                        "diagnostics": self.diagnostics,
                        "noise_source": self.noise_source,
                    }
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        files["extraction"] = provenance_path
        if config.output.save_simulated:
            simulation = self.simulate()
            if simulation is not None:
                simulation_path = Path(output_dir) / "simulated_region.npz"
                np.savez_compressed(
                    simulation_path, simulated=simulation, residual=self._data - simulation
                )
                files["simulation"] = simulation_path
        return MappingProxyType(files)


def _json_metadata(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        result = {item.name: _json_metadata(getattr(value, item.name)) for item in fields(value)}
    elif isinstance(value, Mapping):
        result = {str(key): _json_metadata(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_json_metadata(item) for item in value]
    elif isinstance(value, slice):
        result = {"start": value.start, "stop": value.stop, "step": value.step}
    elif isinstance(value, np.ndarray):
        result = value.tolist()
    elif isinstance(value, np.generic):
        result = value.item()
    elif isinstance(value, Path):
        result = str(value)
    elif isinstance(value, bytes):
        result = {"encoding": "hex", "data": value.hex()}
    elif hasattr(value, "model_dump"):
        result = value.model_dump(mode="json")
    else:
        result = value
    return result


def _project_peaks(outcome, prepared) -> tuple[PeakEstimate, ...]:
    estimates = []
    decisions = {d.seed.id: d for d in prepared.seeds if d.included}
    for cluster in outcome.clusters:
        evaluation = cluster.analytical_evaluation
        for row, name in enumerate(cluster.peak_names):
            params = tuple(
                p
                for p in cluster.final_nonlinear_parameters
                if ParameterId.from_name(p.name).peak_name == name
            )
            positions = {}
            for axis in prepared.extraction.axes:
                matches = [p.value for p in params if p.name == f"{name}.{axis.label}.cs"]
                if matches:
                    positions[axis.external_axis] = matches[0]
            decision = decisions[name]
            errors = None
            if evaluation is not None:
                # Never broadcast the shared amplitude covariance across series.
                actual_errors = evaluation.amplitude_standard_errors
                if actual_errors.shape == evaluation.amplitudes.shape:
                    errors = tuple(float(value) for value in actual_errors[row])
            estimates.append(
                PeakEstimate(
                    name,
                    MappingProxyType(positions),
                    params,
                    tuple(float(x) for x in evaluation.amplitudes[row])
                    if evaluation is not None
                    else None,
                    errors,
                    cluster.classification.value,
                    decision.seed,
                    decision.in_analysis_region,
                )
            )
    return tuple(estimates)


def fit_spectrum(
    source: SpectrumInput,
    *,
    peaks: Sequence[PeakSeed] | None = None,
    peaklist_path: str | Path | None = None,
    region_ppm: Region | None = None,
    config: PeakFitConfig | None = None,
    output_dir: str | Path | None = None,
    auto_pick: bool = False,
    halo_points: int = 0,
    noise: float | None = None,
    noise_region_ppm: Region | None = None,
    optimizer: str = "varpro",
    workers: int = 1,
    max_memory_bytes: int | None = None,
    progress: Callable[[Any], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> SpectrumFitResult:
    """Fit an in-memory spectrum through PeakFit with no mandatory files or GUI."""
    if optimizer not in {"varpro", "basin_hopping"}:
        raise ValueError("optimizer must be varpro or basin_hopping.")
    prepared = prepare_spectrum(
        source,
        peaks=peaks,
        peaklist_path=peaklist_path,
        region_ppm=region_ppm,
        config=config,
        auto_pick=auto_pick,
        halo_points=halo_points,
        noise=noise,
        noise_region_ppm=noise_region_ppm,
        workers=workers,
        max_memory_bytes=max_memory_bytes,
        cancel=cancel,
        progress=progress,
    )

    def callback(event: Any) -> None:
        _check_cancel(cancel)
        if progress is not None:
            progress(event)
        _check_cancel(cancel)

    run = fit_data(
        prepared.data,
        prepared.config,
        optimizer=optimizer,
        workers=workers,
        progress_callback=callback,
    )
    _check_cancel(cancel)
    if run.simulation_snapshot is None:
        raise RuntimeError("Completed fit has no final model snapshot.")
    result = SpectrumFitResult(
        run.final_outcome,
        _project_peaks(run.final_outcome, prepared),
        prepared.extraction,
        prepared.seeds,
        prepared.models,
        prepared.diagnostics,
        prepared.data.noise_source,
        run.simulation_snapshot,
        prepared.data.spectra.data,
        prepared.config.model_dump_json(),
    )
    _check_cancel(cancel)
    if output_dir is not None:
        result.export(output_dir)
    return result


class FitOptions(TypedDict, total=False):
    """Keyword options shared by fit_spectrum and fit_file."""

    peaks: Sequence[PeakSeed] | None
    peaklist_path: str | Path | None
    region_ppm: Region | None
    config: PeakFitConfig | None
    output_dir: str | Path | None
    auto_pick: bool
    halo_points: int
    noise: float | None
    noise_region_ppm: Region | None
    optimizer: str
    workers: int
    max_memory_bytes: int | None
    progress: Callable[[Any], None] | None
    cancel: Callable[[], bool] | None


def fit_file(
    path: str | Path,
    *,
    direct_axis: int | None = None,
    series_axis: int | None = None,
    series: SeriesAxis | None = None,
    processing: Mapping[int, AxisProcessing] | None = None,
    **fit_options: Unpack[FitOptions],
) -> SpectrumFitResult:
    """Read supported processed data and fit using the same array preparation path."""
    source = SpectrumInput.from_file(
        path, direct_axis=direct_axis, series_axis=series_axis, series=series, processing=processing
    )
    return fit_spectrum(source, **fit_options)
