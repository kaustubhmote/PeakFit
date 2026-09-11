from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.io.readers import spectrum as spectrum_reader

if TYPE_CHECKING:
    from pathlib import Path


def _make_bruker_directory(path: Path, binary_name: str = "2rr") -> Path:
    path.mkdir(parents=True)
    (path / "procs").touch()
    (path / binary_name).touch()
    return path


def test_resolve_bruker_component_processing_directory_and_experiment(tmp_path: Path) -> None:
    processed = _make_bruker_directory(tmp_path / "experiment" / "pdata" / "1")

    assert spectrum_reader.resolve_spectrum_path(processed / "2rr") == processed
    assert spectrum_reader.resolve_spectrum_path(processed) == processed
    assert spectrum_reader.resolve_spectrum_path(tmp_path / "experiment") == processed


def test_resolve_bruker_requires_procno_when_selection_is_ambiguous(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    _make_bruker_directory(experiment / "pdata" / "2")
    _make_bruker_directory(experiment / "pdata" / "3")

    with pytest.raises(ValueError, match="Multiple processed Bruker spectra"):
        spectrum_reader.resolve_spectrum_path(experiment)


def test_read_processed_bruker_spectrum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    processed = _make_bruker_directory(tmp_path / "experiment" / "pdata" / "1")
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    dic = {
        "proc2s": {"FT_mod": 5, "WDW": 0, "PHC1": 180, "FTSIZE": 8},
        "procs": {"FT_mod": 6, "WDW": 4, "PHC1": 0, "FTSIZE": 16},
        "acqu2s": {"TD": 6},
        "acqus": {"TD": 8, "AQ": 0.02},
    }
    udic = {
        0: {"size": 3, "sw": 3000.0, "obs": 60.0, "car": 7200.0, "label": "<15N>"},
        1: {"size": 4, "sw": 8000.0, "obs": 800.0, "car": 3760.0, "label": "<1H>"},
    }
    monkeypatch.setattr(spectrum_reader.bruker, "read_pdata", lambda _path: (dic, data))
    monkeypatch.setattr(
        spectrum_reader.bruker,
        "guess_udic",
        lambda _dic, _data, strip_fake: udic if strip_fake else None,
    )

    spectra = spectrum_reader.read_spectra(processed)

    assert spectra.data.shape == (1, 3, 4)
    assert spectra.pseudo_dim_added is True
    assert spectra.z_values.tolist() == [0]
    assert [param.nucleus for param in spectra.spectral_params] == ["15N", "1H"]
    assert [param.label for param in spectra.spectral_params] == ["F2", "F3"]
    assert spectra.spectral_params[0].p180 is True
    assert get_shape_names(FitConfig(), spectra) == ["no_apod", "pvoigt"]


def test_bruker_nontransformed_first_axis_is_the_pseudo_dimension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    processed = _make_bruker_directory(tmp_path / "experiment" / "pdata" / "1", "3rrr")
    data = np.zeros((5, 3, 4), dtype=np.float32)
    dic = {
        "proc3s": {"FT_mod": 0},
        "proc2s": {"FT_mod": 5, "WDW": 0, "FTSIZE": 8},
        "procs": {"FT_mod": 6, "WDW": 0, "FTSIZE": 16},
        "acqu2s": {"TD": 6},
        "acqus": {"TD": 8, "AQ": 0.02},
    }
    udic = {
        0: {"size": 5, "sw": 1.0, "obs": 1.0, "car": 0.0, "label": ""},
        1: {"size": 3, "sw": 3000.0, "obs": 60.0, "car": 7200.0, "label": "15N"},
        2: {"size": 4, "sw": 8000.0, "obs": 800.0, "car": 3760.0, "label": "1H"},
    }
    monkeypatch.setattr(spectrum_reader.bruker, "read_pdata", lambda _path: (dic, data))
    monkeypatch.setattr(
        spectrum_reader.bruker,
        "guess_udic",
        lambda _dic, _data, strip_fake: udic if strip_fake else None,
    )

    spectra = spectrum_reader.read_spectra(processed)

    assert spectra.data.shape == (5, 3, 4)
    assert spectra.pseudo_dim_added is False
    assert spectra.z_values.tolist() == [0, 1, 2, 3, 4]
    assert spectra.n_spectral_dims == 2
