from click.utils import strip_ansi
from typer.testing import CliRunner

from peakfit.cli.app import app


def rendered_help(*args: str) -> str:
    result = CliRunner().invoke(app, [*args, "--help"])

    assert result.exit_code == 0, result.output
    return strip_ansi(result.output)


def test_root_help_uses_pseudo_nd_terminology() -> None:
    help_text = rendered_help()

    assert "pseudo-ND NMR spectra" in help_text
    assert "pseudo-3D" not in help_text


def test_fit_help_describes_plane_values_without_renaming_the_option() -> None:
    help_text = rendered_help("fit")

    assert "pseudo-ND NMR spectrum" in help_text
    assert "--z-values" in help_text
    assert "Plane values file" in help_text
    assert "Z-dimension" not in help_text


def test_fit_help_exposes_tilted_peak_geometry() -> None:
    help_text = rendered_help("fit")
    assert "--tilted" in help_text
    assert "Fit rotated" in help_text
    assert "two-dimensional" in help_text


def test_plot_spectrum_help_mentions_bruker_input() -> None:
    help_text = rendered_help("plot", "spectrum")

    assert "NMRPipe file or processed Bruker spectrum" in help_text


def test_fit_help_exposes_recorded_window_option() -> None:
    result = CliRunner().invoke(app, ["fit", "--help"], env={"COLUMNS": "180"})
    assert result.exit_code == 0
    help_text = strip_ansi(result.output)
    assert "--include-window-functions" in help_text
    assert "acquisition windows" in help_text
