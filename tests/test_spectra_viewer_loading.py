from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.cli.commands import plot as plot_command
from peakfit.io.readers import spectrum as spectrum_reader
from peakfit.plot import spectra_viewer
from peakfit.plot.spectra_viewer import NMRData

if TYPE_CHECKING:
    from pathlib import Path


def _stub_bruker_2rr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, list[str]]:
    processed = tmp_path / "experiment" / "pdata" / "1"
    processed.mkdir(parents=True)
    (processed / "procs").touch()
    binary = processed / "2rr"
    binary.touch()

    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    dic = {
        "proc2s": {"FT_mod": 5, "WDW": 0, "FTSIZE": 8},
        "procs": {"FT_mod": 6, "WDW": 0, "FTSIZE": 16},
        "acqu2s": {"TD": 6},
        "acqus": {"TD": 8, "AQ": 0.02},
    }
    udic = {
        0: {"size": 3, "sw": 3000.0, "obs": 60.0, "car": 7200.0, "label": "15N"},
        1: {"size": 4, "sw": 8000.0, "obs": 800.0, "car": 3760.0, "label": "1H"},
    }
    read_paths: list[str] = []

    def _read_pdata(path: str):
        read_paths.append(path)
        return dic, data

    monkeypatch.setattr(spectrum_reader.bruker, "read_pdata", _read_pdata)
    monkeypatch.setattr(
        spectrum_reader.bruker,
        "guess_udic",
        lambda _dic, _data, strip_fake: udic if strip_fake else None,
    )
    return binary, read_paths


def test_nmr_data_loads_direct_bruker_2rr_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary, read_paths = _stub_bruker_2rr(monkeypatch, tmp_path)

    loaded = NMRData.from_file(str(binary))

    assert read_paths == [str(binary.parent)]
    assert loaded.filename == str(binary)
    assert loaded.data.shape == (1, 3, 4)
    assert loaded.xlim == pytest.approx((9.7, 2.2))
    assert loaded.ylim == pytest.approx((145.0, 111.6666666667))


def test_plot_spectrum_command_loads_bruker_2rr_with_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary, read_paths = _stub_bruker_2rr(monkeypatch, tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    captured: dict[str, object] = {}

    def _init_reconstructor(path: Path | None) -> None:
        captured["results"] = path

    def fake_serve(data, _reconstructor, _peaks, **kwargs):
        captured["data"] = data
        captured["browser"] = kwargs["open_browser"]
        captured["port"] = kwargs["port"]
        kwargs["on_ready"]("http://127.0.0.1:8765/session/")

    monkeypatch.setattr(plot_command, "_init_reconstructor", _init_reconstructor)
    monkeypatch.setattr(spectra_viewer, "serve_viewer", fake_serve)

    result = CliRunner().invoke(
        app,
        [
            "plot",
            "spectrum",
            "--spectrum",
            str(binary),
            "--results",
            str(results),
            "--no-browser",
            "--port",
            "8765",
        ],
    )

    assert result.exit_code == 0, result.output
    assert read_paths == [str(binary.parent)]
    assert captured["results"] == results
    assert isinstance(captured["data"], NMRData)
    assert captured["browser"] is False
    assert captured["port"] == 8765
    assert "http://127.0.0.1:8765/session/" in result.output
