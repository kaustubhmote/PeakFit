"""Quadrature input identification, validation, and CLI wiring."""

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from nmrglue.fileio import fileiobase, pipe
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.fit.fitting import load_data
from peakfit.io.readers.spectrum import read_spectra
from peakfit.shared.exceptions import DataIOError

if TYPE_CHECKING:
    from pathlib import Path


def write_plane(path: Path, value: float, *, sw=80.0, points=16) -> None:
    udic = fileiobase.create_blank_udic(2)
    for axis, center in enumerate([8.0, 4.5]):
        udic[axis].update(
            size=points, sw=sw, obs=600.0, car=center * 600, complex=False, time=False, freq=True
        )
    dic = pipe.create_dic(udic)
    pipe.write(str(path), dic, np.full((points, points), value, dtype=np.float32), overwrite=True)


@pytest.mark.parametrize("components", [("ir",), ("ri", "ii"), ("ri", "ir", "ii")])
def test_manually_supplied_nmrpipe_component_planes(tmp_path, components):
    real = tmp_path / "real.ft2"
    write_plane(real, 0)
    paths = {}
    for index, component in enumerate(components, 1):
        path = tmp_path / f"{component}.ft2"
        write_plane(path, index)
        paths[component] = path
    spectra = read_spectra(real, use_imaginary=True, imaginary_paths=paths)
    expected = ("rr", *(component for component in ("ri", "ir", "ii") if component in paths))
    assert spectra.quadrature_components == expected
    for component, plane in zip(expected[1:], spectra.data[1:], strict=True):
        np.testing.assert_array_equal(plane, components.index(component) + 1)
    with pytest.raises(ValueError, match="require --use-imaginary"):
        read_spectra(real, imaginary_paths=paths)


@pytest.mark.parametrize("change", [{"sw": 100.0}, {"points": 32}])
def test_component_grid_mismatches_are_rejected(tmp_path, change):
    real, imaginary = tmp_path / "real.ft2", tmp_path / "imaginary.ft2"
    write_plane(real, 0)
    write_plane(imaginary, 1, **change)
    with pytest.raises(ValueError, match="grid"):
        read_spectra(real, use_imaginary=True, imaginary_paths={"ir": imaginary})


def test_quadrature_rejects_missing_component_files_and_series_options(tmp_path):
    real = tmp_path / "real.ft2"
    write_plane(real, 0)
    with pytest.raises(ValueError, match="not a file"):
        read_spectra(real, use_imaginary=True, imaginary_paths={"ir": tmp_path / "missing"})
    with pytest.raises(ValueError, match="at least one imaginary"):
        read_spectra(real, use_imaginary=True)
    with pytest.raises(ValueError, match="--z-values or --exclude"):
        read_spectra(real, use_imaginary=True, exclude_list=[1])
    with pytest.raises(DataIOError, match="requires --lineshape dqfcosy"):
        load_data(real, None, None, PeakFitConfig(fitting=FitConfig(use_imaginary=True)))


def test_cli_flag_and_manual_components_override_config(tmp_path, monkeypatch):
    real, imaginary = tmp_path / "real.ft2", tmp_path / "imaginary.ft2"
    write_plane(real, 0)
    write_plane(imaginary, 1)
    peaklist = tmp_path / "centers.list"
    peaklist.write_text("A23 8.0 4.5\n")
    config = tmp_path / "settings.toml"
    config.write_text('[fitting]\nlineshape = "dqfcosy"\n')
    captured = {}

    def load(*, fit_config, **_kwargs):
        captured["config"] = fit_config
        return SimpleNamespace(peaks=[])

    monkeypatch.setattr("peakfit.cli.commands.fit._load_fit_data", load)
    monkeypatch.setattr(
        "peakfit.cli.commands.fit.run_fit",
        lambda _data, _config, output_dir, **_kwargs: SimpleNamespace(
            output_dir=output_dir, spectra=SimpleNamespace()
        ),
    )
    monkeypatch.setattr("peakfit.cli.commands.fit.write_fit_run_outputs", lambda *_args: None)
    result = CliRunner().invoke(
        app,
        [
            "fit",
            str(real),
            str(peaklist),
            "--config",
            str(config),
            "--use-imaginary",
            "--imaginary",
            f"ir={imaginary}",
            "--output",
            str(tmp_path / "results"),
            "--headless",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["config"].fitting.use_imaginary
    assert captured["config"].fitting.imaginary_paths == {"ir": imaginary}


@pytest.mark.parametrize("argument", ["bad=somewhere", "ir", "ii=", "ir=x,ir=y"])
def test_cli_rejects_unidentified_or_duplicate_components(tmp_path, argument):
    real = tmp_path / "real.ft2"
    write_plane(real, 0)
    args = ["fit", str(real), "--use-imaginary", "--lineshape", "dqfcosy"]
    for item in argument.split(","):
        args.extend(["--imaginary", item])
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "Duplicate imaginary component" in result.output or "Use --imaginary" in result.output
