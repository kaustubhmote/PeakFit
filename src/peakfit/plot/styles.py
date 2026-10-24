"""Validate spectrum styles and translate them to Matplotlib contour options."""

from typing import TYPE_CHECKING, Any

from matplotlib import colormaps
from matplotlib.colors import is_color_like, to_hex

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


def browser_styles(config: PlotConfig) -> dict[str, dict[str, Any]]:
    """Serialize the same signed colors and widths for the browser canvas."""
    contour_options(config)
    styles = {}
    for layer, style in (
        ("experimental", config.experimental),
        ("simulated", config.simulated),
        ("residual", config.difference),
    ):
        palette = (
            [to_hex(colormaps[style.colormap](index / 255)) for index in range(256)]
            if style.colormap is not None
            else None
        )
        styles[layer] = {
            "colors": [to_hex(style.color)] * 2 if palette is None else [palette[-1], palette[0]],
            "linewidth": style.linewidth if style.linewidth is not None else 1.0,
            "colormap": palette,
        }
    return styles
