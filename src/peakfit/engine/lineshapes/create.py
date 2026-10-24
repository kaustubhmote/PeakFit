"""Construct built-in lineshape model instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from peakfit.engine.lineshapes.dqfcosy import DQFCOSY
from peakfit.engine.lineshapes.gaussian.model import Gaussian, GaussianDoublet
from peakfit.engine.lineshapes.lorentzian.model import Lorentzian, LorentzianDoublet
from peakfit.engine.lineshapes.no_apod.model import NoApod, NoApodDoublet
from peakfit.engine.lineshapes.pvoigt.model import PseudoVoigt, PseudoVoigtDoublet
from peakfit.engine.lineshapes.sp1.model import SP1, SP1Doublet
from peakfit.engine.lineshapes.sp2.model import SP2, SP2Doublet

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from peakfit.engine.domain.config import FitConfig
    from peakfit.engine.domain.spectrum import Spectra
    from peakfit.engine.types import Shape


SHAPES: dict[str, Callable[..., Shape]] = {
    "dqfcosy": DQFCOSY,
    "gaussian": Gaussian,
    "gaussian_doublet": GaussianDoublet,
    "lorentzian": Lorentzian,
    "lorentzian_doublet": LorentzianDoublet,
    "no_apod": NoApod,
    "no_apod_doublet": NoApodDoublet,
    "pvoigt": PseudoVoigt,
    "pvoigt_doublet": PseudoVoigtDoublet,
    "sp1": SP1,
    "sp1_doublet": SP1Doublet,
    "sp2": SP2,
    "sp2_doublet": SP2Doublet,
}


def _available_shapes() -> list[str]:
    """Return the available built-in shape names."""
    return sorted(SHAPES)


def _create_shape(
    spectra: Spectra,
    config: FitConfig,
    shape_type: str,
    *,
    peak_name: str,
    center: float,
    dim: int,
) -> Shape:
    """Create a single shape instance for the provided peak."""
    shape_provider = _resolve_shape_provider(shape_type)
    return shape_provider(peak_name, center, spectra, dim, config)


def create_shapes(
    spectra: Spectra,
    config: FitConfig,
    peak_name: str,
    positions: Sequence[float],
    shape_names: Sequence[str],
) -> list[Shape]:
    """Create shapes for each peak dimension using provided names."""
    if len(positions) != len(shape_names):
        msg = "Number of positions and shape names must match"
        raise ValueError(msg)
    if "dqfcosy" in shape_names and (list(shape_names) != ["dqfcosy", "dqfcosy"]):
        raise ValueError("DQF-COSY must be used on both axes of a two-dimensional spectrum.")

    if config.include_window_functions:
        _validate_processing_inputs(spectra, config, shape_names)

    shapes: list[Shape] = []
    for dim, (center, shape_name) in enumerate(zip(positions, shape_names, strict=False), start=1):
        shapes.append(
            _create_shape(
                spectra,
                config,
                shape_name,
                peak_name=peak_name,
                center=center,
                dim=dim,
            )
        )
    return shapes


def _resolve_shape_provider(shape_type: str) -> Callable[..., Shape]:
    try:
        return SHAPES[shape_type]
    except KeyError as exc:
        available = ", ".join(_available_shapes())
        msg = f"Unknown lineshape '{shape_type}'. Available: {available}"
        raise ValueError(msg) from exc


def _validate_processing_inputs(
    spectra: Spectra,
    config: FitConfig,
    names: Sequence[str],
) -> None:
    if list(names) == ["dqfcosy", "dqfcosy"]:
        return  # DQF-COSY already owns a window-aware forward model.
    supported = {
        "gaussian",
        "lorentzian",
        "pvoigt",
        "gaussian_doublet",
        "lorentzian_doublet",
        "pvoigt_doublet",
    }
    if set(names) - supported:
        raise ValueError(
            "--include-window-functions requires auto, gaussian, lorentzian, "
            "or pvoigt intrinsic shapes, including doublets."
        )
    if config.use_imaginary:
        raise ValueError("Window-aware intrinsic fitting does not support imaginary components.")
    if spectra.n_spectral_dims not in {1, 2}:
        raise ValueError("--include-window-functions supports one or two spectral dimensions.")
    for spec in spectra.spectral_params:
        if spec.processing_window is None:
            detail = spec.processing_window_error or "missing acquisition/window metadata"
            raise ValueError(f"Cannot include window functions on {spec.label}: {detail}.")


__all__ = ["create_shapes"]
