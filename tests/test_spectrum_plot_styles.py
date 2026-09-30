from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from pydantic import ValidationError
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.cli.commands import plot as plot_command
from peakfit.engine.domain.config import PeakFitConfig, PlotConfig
from peakfit.io.config import generate_default_config, load_config, save_config
from peakfit.plot.qt_core import QApplication
from peakfit.plot.spectra_viewer import NMRData, SpectraViewer
from peakfit.plot.styles import contour_options

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    instance = QApplication.instance() or QApplication([])
    assert isinstance(instance, QApplication)
    return instance


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


@pytest.mark.gui
@pytest.mark.parametrize("configured", [False, True])
def test_viewer_renders_layer_styles_and_preserves_signed_contours(
    qt_app: QApplication, configured: bool
) -> None:
    assert qt_app is not None
    config = PlotConfig()
    if configured:
        config = PlotConfig.model_validate(
            {
                "experimental": {"color": "#224466", "linewidth": 0.5},
                "simulated": {"color": "#aa4422", "linewidth": 2.0},
                "difference": {"colormap": "RdBu_r", "linewidth": 0.75},
            }
        )

    x, y = np.meshgrid(np.linspace(-3, 3, 32), np.linspace(-3, 3, 32))
    plane = 100 * (np.exp(-((x - 1) ** 2 + y**2)) - np.exp(-((x + 1) ** 2 + y**2)))
    data_exp = NMRData("experimental", {}, plane[None], (10, 0), (140, 100))
    data_sim = NMRData("simulated", {}, 0.8 * plane[None], (10, 0), (140, 100))
    viewer = SpectraViewer(data_exp, data_sim, None, plot_config=config)
    try:
        assert len(viewer.backend.ax.collections) == 2
        viewer.control_widget.checkboxes["difference"].setChecked(True)
        contours = viewer.backend.ax.collections
        assert len(contours) == 3
        for contour, style in zip(
            contours, (config.experimental, config.simulated, config.difference), strict=True
        ):
            np.testing.assert_allclose(contour.levels, -contour.levels[::-1])
            assert len(contour.levels) == 50
            default_width = rcParams["contour.linewidth"] or rcParams["lines.linewidth"]
            np.testing.assert_allclose(contour.get_linewidths(), style.linewidth or default_width)
            if style.colormap is None:
                expected_colors = np.tile(to_rgba(style.color, 0.7), (len(contour.levels), 1))
                np.testing.assert_allclose(contour.get_edgecolors(), expected_colors)
            else:
                assert contour.cmap.name == style.colormap
                assert not np.allclose(contour.get_edgecolors()[0], contour.get_edgecolors()[-1])
        assert viewer.data_diff is not None
        np.testing.assert_array_equal(viewer.data_diff, data_exp.data - data_sim.data)
    finally:
        viewer.close()
