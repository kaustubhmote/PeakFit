"""Validate spectrum styles and translate them to Matplotlib contour options."""

from typing import TYPE_CHECKING, Any

from matplotlib import colormaps
from matplotlib.colors import is_color_like

if TYPE_CHECKING:
    from peakfit.engine.domain.config import PlotConfig


def contour_options(config: PlotConfig) -> dict[str, dict[str, Any]]:
    """Return options for each viewer layer, rejecting invalid colors and colormaps."""
    options: dict[str, dict[str, Any]] = {}
    for name, key, style in (
        ("experimental", "spectrum_exp", config.experimental),
        ("simulated", "spectrum_sim", config.simulated),
        ("difference", "difference", config.difference),
    ):
        if not is_color_like(style.color):
            msg = f"plot.{name}.color: unknown Matplotlib color {style.color!r}"
            raise ValueError(msg)
        if style.colormap is not None and style.colormap not in colormaps:
            msg = f"plot.{name}.colormap: unknown Matplotlib colormap {style.colormap!r}"
            raise ValueError(msg)
        options[key] = {
            "colors": style.color if style.colormap is None else None,
            "cmap": style.colormap,
            "linewidths": style.linewidth,
        }
    return options
