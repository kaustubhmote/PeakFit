from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib.colors import to_hex
from pydantic import ValidationError
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.cli.commands import plot as plot_command
from peakfit.engine.domain.config import PeakFitConfig, PlotConfig
from peakfit.io.config import generate_default_config, load_config, save_config
from peakfit.plot.spectra_viewer import NMRData, SpectrumViewer
from peakfit.plot.styles import browser_styles, contour_options

if TYPE_CHECKING:
    from pathlib import Path


def test_plot_config_defaults_and_toml_roundtrip(tmp_path: Path) -> None:
    generated = PeakFitConfig.model_validate(tomllib.loads(generate_default_config()))
    assert generated.plot == PlotConfig()
    assert contour_options(generated.plot) == {
        "spectrum_exp": {"colors": "C0", "cmap": None, "linewidths": None},
        "spectrum_sim": {"colors": "C1", "cmap": None, "linewidths": None},
        "difference": {"colors": "C2", "cmap": None, "linewidths": None},
    }

    path = tmp_path / "plot.toml"
    path.write_text('[plot.difference]\ncolormap = "RdBu_r"\nlinewidth = 0.75\n')
    config = load_config(path)
    save_config(config, path)
    assert load_config(path) == config
    assert config.plot.experimental.color == "C0"
    assert contour_options(config.plot)["difference"] == {
        "colors": None,
        "cmap": "RdBu_r",
        "linewidths": 0.75,
    }


@pytest.mark.parametrize("linewidth", [0, -1, float("nan"), float("inf")])
def test_invalid_linewidth_is_rejected(linewidth: float) -> None:
    with pytest.raises(ValidationError, match="linewidth"):
        PlotConfig.model_validate({"experimental": {"linewidth": linewidth}})


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"experimental": {"colour": "red"}}, "experimental.colour"),
        ({"experiment": {"color": "red"}}, "experiment"),
        ({"difference": {"colormap": ""}}, "difference.colormap"),
    ],
)
def test_plot_config_rejects_unknown_fields_and_empty_names(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        PlotConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ('color = "unknown-color"', "plot.simulated.color"),
        ('colormap = "unknown-map"', "plot.simulated.colormap"),
        ("linewidth = -1", "plot.simulated.linewidth"),
        ('linewidth = "unfinished', "loading plot configuration"),
    ],
)
def test_bad_plot_config_fails_before_spectrum_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: str, message: str
) -> None:
    spectrum = tmp_path / "spectrum.ft2"
    spectrum.touch()
    config = tmp_path / "plot.toml"
    config.write_text(f"[plot.simulated]\n{settings}\n")

    def unexpected_load(_path: Path) -> None:
        pytest.fail("Invalid plot configuration must fail before loading the spectrum")

    monkeypatch.setattr(plot_command, "_load_spectrum", unexpected_load)
    result = CliRunner().invoke(
        app, ["plot", "spectrum", "-s", str(spectrum), "--config", str(config)]
    )
    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize("configured", [False, True])
def test_browser_styles_preserve_signed_contour_geometry(configured: bool) -> None:
    config = (
        PlotConfig.model_validate(
            {
                "experimental": {"color": "#224466", "linewidth": 0.5},
                "simulated": {"color": "#aa4422", "linewidth": 2.0},
                "difference": {"colormap": "RdBu_r", "linewidth": 0.75},
            }
        )
        if configured
        else None
    )
    x, y = np.meshgrid(np.linspace(-3, 3, 32), np.linspace(-3, 3, 32))
    plane = 100 * (np.exp(-((x - 1) ** 2 + y**2)) - np.exp(-((x + 1) ** 2 + y**2)))
    data = NMRData("experimental", {}, plane[None], (10, 0), (140, 100))
    viewer = SpectrumViewer(data, plot_config=config)
    unstyled = SpectrumViewer(data)
    assert viewer.contours(0, "experimental", [1, 2]) == unstyled.contours(
        0, "experimental", [1, 2]
    )
    styles = viewer.metadata()["styles"]
    if config is None:
        assert styles == {}
    else:
        assert styles == browser_styles(config)
        assert styles["experimental"]["colors"] == [to_hex(config.experimental.color)] * 2
        assert styles["experimental"]["linewidth"] == 0.5
        palette = styles["residual"]["colormap"]
        assert len(palette) == 256
        assert palette[0] != palette[-1]
