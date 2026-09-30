"""Configuration and output setup for the fit CLI command."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, get_args

import typer

from peakfit.engine.domain.config import (
    ClusterConfig,
    FitConfig,
    LineshapeName,
    OutputConfig,
    OutputFormat,
    PeakFitConfig,
)
from peakfit.io.config import load_config
from peakfit.shared.constants import BASIN_HOPPING_NITER

if TYPE_CHECKING:
    from peakfit.engine.domain.peaks import Peak

VALID_OUTPUT_FORMATS = get_args(OutputFormat)
VALID_OPTIMIZERS = ("varpro", "basin_hopping")


def build_fit_config(
    config: Path | None,
    output: Path | None,
    formats: list[str] | None,
    headless: bool | None,
    noise: float | None,
    contour_level: float | None,
    lineshape: LineshapeName,
    tilted: bool,
    refine: int,
    fixed: bool,
    jx: bool,
    phx: bool,
    phy: bool,
    exclude: list[int] | None,
    optimizer: str,
    include_window_functions: bool = False,
    use_imaginary: bool = False,
    imaginary: list[str] | None = None,
) -> PeakFitConfig:
    """Build fit config from CLI options or a TOML config file."""
    validate_optimizer(optimizer)
    normalized_formats = normalize_output_formats(formats)
    imaginary_paths = _parse_imaginary_paths(imaginary)

    if config is not None:
        cfg = load_config(config)
        if output:
            cfg.output.directory = output
        if normalized_formats is not None:
            cfg.output.formats = normalized_formats
        if headless is not None:
            cfg.output.headless = headless
        if noise is not None:
            cfg.noise_level = noise
        if contour_level is not None:
            cfg.clustering.contour_level = contour_level
        cfg.fitting.include_window_functions |= include_window_functions
        cfg.fitting.use_imaginary = cfg.fitting.use_imaginary or use_imaginary
        cfg.fitting.imaginary_paths = (
            imaginary_paths if imaginary is not None else cfg.fitting.imaginary_paths
        )
        return cfg

    fit_phase = []
    if phx:
        fit_phase.append("F3")
    if phy:
        fit_phase.append("F2")

    default_formats: list[OutputFormat] = ["json", "csv"]
    output_config = OutputConfig(
        formats=normalized_formats if normalized_formats is not None else default_formats,
        headless=headless if headless is not None else False,
    )
    if output:
        output_config.directory = output

    cfg = PeakFitConfig(
        fitting=FitConfig(
            lineshape=lineshape,
            tilted=tilted,
            include_window_functions=include_window_functions,
            refine_iterations=refine,
            fix_positions=fixed,
            fit_j_coupling=jx,
            fit_phase=fit_phase,
            use_imaginary=use_imaginary,
            imaginary_paths=imaginary_paths,
        ),
        clustering=ClusterConfig(contour_level=contour_level),
        output=output_config,
        noise_level=noise,
        exclude_planes=exclude or [],
    )

    if optimizer == "basin_hopping":
        cfg.fitting.max_iterations = BASIN_HOPPING_NITER

    return cfg


def _parse_imaginary_paths(items: list[str] | None) -> dict[Literal["ri", "ir", "ii"], Path]:
    paths: dict[Literal["ri", "ir", "ii"], Path] = {}
    for item in items or []:
        component, separator, path = item.partition("=")
        if not separator or component not in {"ri", "ir", "ii"} or not path:
            raise typer.BadParameter("Use --imaginary ri=PATH, ir=PATH, or ii=PATH.")
        key = cast("Literal['ri', 'ir', 'ii']", component)
        if key in paths:
            raise typer.BadParameter(f"Duplicate imaginary component: {component}.")
        paths[key] = Path(path).expanduser().resolve()
    return paths


def validate_optimizer(optimizer: str) -> None:
    """Reject unsupported optimizer names before data loading starts."""
    if optimizer in VALID_OPTIMIZERS:
        return

    choices = ", ".join(VALID_OPTIMIZERS)
    raise typer.BadParameter(f"Unknown optimizer: {optimizer}. Choose from: {choices}.")


def normalize_output_formats(formats: list[str] | None) -> list[OutputFormat] | None:
    """Validate and normalize repeated --format values."""
    if not formats:
        return None

    allowed = set(VALID_OUTPUT_FORMATS)
    normalized = [fmt.lower() for fmt in formats]
    invalid = sorted({fmt for fmt in normalized if fmt not in allowed})
    if invalid:
        allowed_text = ", ".join(VALID_OUTPUT_FORMATS)
        invalid_text = ", ".join(invalid)
        raise typer.BadParameter(
            f"Unknown output format(s): {invalid_text}. Choose from: {allowed_text}."
        )

    deduped = list(dict.fromkeys(normalized))
    return cast("list[OutputFormat]", deduped)


def resolve_output_dir(fit_config: PeakFitConfig) -> Path:
    """Resolve output directory and apply timestamp policy."""
    base = fit_config.output.directory or Path("./")
    output_dir = Path(base)
    if fit_config.output.include_timestamp:
        timestamp = datetime.datetime.now(datetime.UTC).astimezone().strftime("%Y%m%d_%H%M%S")
        if str(output_dir) == ".":
            output_dir = Path(f"output_{timestamp}")
        else:
            output_dir = output_dir / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    fit_config.output.directory = output_dir
    fit_config.output.include_timestamp = False
    return output_dir


def write_autopicked_peaklist(output_dir: Path, peaks: list[Peak]) -> Path:
    """Write auto-detected peaks to Sparky-like list format for reproducibility."""
    peaklist_dir = output_dir / "metadata"
    peaklist_dir.mkdir(parents=True, exist_ok=True)
    peaklist_path = peaklist_dir / "autopicked.list"

    columns = [f"w{i + 1}" for i in range(len(peaks[0].positions))] if peaks else []
    header = "Assignment" if not columns else f"Assignment {' '.join(columns)}"

    lines = [header]
    for peak in peaks:
        positions = " ".join(f"{float(pos):.6f}" for pos in peak.positions)
        lines.append(f"{peak.name} {positions}")

    peaklist_path.write_text("\n".join(lines) + "\n")
    return peaklist_path
