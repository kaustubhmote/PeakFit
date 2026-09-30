from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from peakfit.engine.domain.config import DQFCOSYWindow, FitConfig
from peakfit.engine.domain.spectrum import get_shape_names
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.lineshapes.dqfcosy import DQFCOSY, antiphase_profile
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


@pytest.mark.parametrize(("wdw", "power"), [(0, 0), (1, 0), (3, 1), (4, 2)])
@pytest.mark.parametrize(("ssb", "offset"), [(0, 0), (1, 0), (1.5, 0), (2, 0.5), (4, 0.25)])
def test_bruker_window_matches_independent_topspin_fid(wdw, power, ssb, offset):
    acquisition = {"TD": 128, "SW_h": 1000.0, "AQ_mod": 3}
    processing = {"WDW": wdw, "SSB": ssb, "LB": 1.5, "SI": 256}
    window = spectrum_reader._bruker_dqfcosy_window(acquisition, processing, direct=True)
    assert window.points == 64
    assert window.acquisition_time == pytest.approx(0.064)
    assert window.sine_power == power
    # LB belongs to EM; inactive saved LB/SSB must not modify another window.
    times = np.arange(64) / 1000
    weights = np.exp(-np.pi * (6.0 + (1.5 if wdw == 1 else 0)) * times)
    if power:
        weights *= np.sin(np.pi * (offset + (1 - offset) * times / 0.064)) ** power
    offsets = np.linspace(-30, 30, 151)
    fid = 2j * np.sin(np.pi * 4.2 * times) * weights
    expected = np.exp(-2j * np.pi * np.outer(offsets, times)) @ fid / 64
    actual = antiphase_profile(offsets, np.array(6.0), np.array(4.2), window)[0]
    np.testing.assert_allclose(actual, expected, atol=2e-13)


@pytest.mark.parametrize(("fnmode", "mc2"), [(4, 0), (5, 0), (6, 0), (0, 3), (0, 4), (0, 5)])
def test_bruker_indirect_timing_uses_quadrature_and_acquisition_width(fnmode, mc2):
    # Acquisition width is independent of the cropped processing width.
    acquisition = {"TD": 512, "FnMODE": fnmode, "SW": 10.0, "SFO1": 600.0}
    processing = {"WDW": 4, "SSB": 2, "SI": 1024, "TDeff": 256, "SW_p": 1500, "MC2": mc2}
    window = spectrum_reader._bruker_dqfcosy_window(acquisition, processing, direct=False)
    assert window.points == 128
    assert window.acquisition_time == pytest.approx(128 / 6000)
    assert window.sine_offset == 0.5
    assert window.sine_end == pytest.approx(0.5 + 0.5 * 127 / 256)


def test_bruker_tppi_retains_real_samples_at_half_dwell():
    acquisition = {"TD": 256, "FnMODE": 3, "SW_h": 4000}
    window = spectrum_reader._bruker_dqfcosy_window(
        acquisition, {"WDW": 3, "SSB": 0, "SI": 512}, direct=False
    )
    assert window.points == 256
    assert window.acquisition_time == pytest.approx(256 / 8000)


def test_bruker_si_can_truncate_but_zero_filling_does_not_extend_timing():
    acquisition = {"TD": 2048, "AQ_mod": 3, "SW_h": 8000, "AQ": 100.0}
    truncated = spectrum_reader._bruker_dqfcosy_window(
        acquisition, {"WDW": 0, "SI": 256}, direct=True
    )
    zero_filled = spectrum_reader._bruker_dqfcosy_window(
        acquisition, {"WDW": 0, "SI": 8192, "TDeff": 0}, direct=True
    )
    assert truncated.points == 256
    assert truncated.acquisition_time == pytest.approx(256 / 8000)
    assert zero_filled.points == 1024
    assert zero_filled.acquisition_time == pytest.approx(1024 / 8000)


@pytest.mark.parametrize("components", [(), ("ri",), ("ir", "ii"), ("ri", "ir", "ii")])
def test_nmrglue_bruker_dictionaries_supply_both_windows_without_config(tmp_path, components):
    experiment = tmp_path / "cosy"
    processed = _make_bruker_directory(experiment / "pdata" / "1")
    acquisition = {"TD": 2048, "SW_h": 8000.0, "SFO1": 600.004, "AQ_mod": 3, "NUC1": "<1H>"}
    indirect = {"TD": 512, "SW": 10.0, "SFO1": 600.0, "FnMODE": 5, "NUC1": "<1H>"}
    procs = {
        "SI": 32,
        "FTSIZE": 32,
        "XDIM": 32,
        "SF": 600.0,
        "WDW": 4,
        "SSB": 2.0,
        "TDeff": 64,
        "FT_mod": 6,
        "NC_proc": 0,
        "BYTORDP": 0,
        "DTYPP": 0,
    }
    proc2s = {**procs, "WDW": 3, "SSB": 4.0, "MC2": 4}
    for path, parameters in [
        (experiment / "acqus", acquisition),
        (experiment / "acqu2s", indirect),
        (processed / "procs", procs),
        (processed / "proc2s", proc2s),
    ]:
        spectrum_reader.bruker.write_jcamp(
            {"_comments": [], "_coreheader": [], **parameters}, str(path), overwrite=True
        )
    (processed / "2rr").write_bytes(np.zeros((32, 32), dtype="<i4").tobytes())
    for index, component in enumerate(components, 1):
        (processed / f"2{component}").write_bytes(np.full((32, 32), index, dtype="<i4").tobytes())
    spectra = spectrum_reader.read_spectra(processed / "2rr", use_imaginary=bool(components))
    assert spectra.data.shape == (1 + len(components), 32, 32)
    assert spectra.quadrature_components == (("rr", *components) if components else ())
    for index in range(1, 1 + len(components)):
        np.testing.assert_array_equal(spectra.data[index], index)
    if components:
        custom = processed / "custom_component.bin"
        custom.write_bytes((processed / f"2{components[0]}").read_bytes())
        manual = spectrum_reader.read_spectra(
            processed, use_imaginary=True, imaginary_paths={"ri": custom}
        )
        assert manual.quadrature_components == ("rr", "ri")
        assert manual.data.shape[0] == 2  # Explicit input replaces sibling discovery.
        np.testing.assert_array_equal(manual.data[1], 1)
    else:
        with pytest.raises(ValueError, match="at least one imaginary component"):
            spectrum_reader.read_spectra(processed, use_imaginary=True)
    config = FitConfig(lineshape="dqfcosy")
    shapes = create_shapes(spectra, config, "A23", [8.0, 4.5], get_shape_names(config, spectra))
    assert all(key in spectra.dic for key in ("acqus", "acqu2s", "procs", "proc2s"))
    assert isinstance(shapes[0], DQFCOSY)
    assert isinstance(shapes[1], DQFCOSY)
    assert shapes[0].window.sine_power == 1
    assert shapes[0].window.sine_offset == 0.25
    assert shapes[0].window.acquisition_time == pytest.approx(32 / 6000)
    assert shapes[1].window.sine_power == 2
    assert shapes[1].window.acquisition_time == pytest.approx(32 / 8000)
    override = DQFCOSYWindow(acquisition_time=0.1, points=200, lb_hz=2)
    config.dqfcosy_windows = {"F2": override}
    overridden = create_shapes(spectra, config, "A23", [8.0, 4.5], ["dqfcosy", "dqfcosy"])
    assert isinstance(overridden[0], DQFCOSY)
    assert isinstance(overridden[1], DQFCOSY)
    assert overridden[0].window == override
    assert overridden[1].window == shapes[1].window


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"WDW": 2}, "WDW"),
        ({"WDW": 1, "LB": -1}, "lb_hz"),
        ({"TDoff": 2}, "TDoff"),
        ({"ME_mod": 1}, "ME_mod"),
        ({"TDeff": 3}, "complete real/imaginary pairs"),
    ],
)
def test_unsupported_bruker_processing_has_actionable_error(change, reason):
    with pytest.raises(ValueError, match=reason):
        spectrum_reader._bruker_dqfcosy_window(
            {"TD": 512, "SW_h": 8000, "AQ_mod": 3},
            {"WDW": 4, "SSB": 2, "SI": 256, **change},
            direct=True,
        )


@pytest.mark.parametrize(
    "acquisition",
    [
        {"TD": 3988, "SW_h": 100000.0},
        {"TD": 3988, "SW": 100000.0 / 700.23, "SFO1": 700.23},
    ],
)
def test_bruker_cropped_spectrum_preserves_acquisition_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, acquisition: dict
) -> None:
    processed = _make_bruker_directory(tmp_path / "experiment" / "pdata" / "1")
    data = np.zeros((16, 256), dtype=np.float32)
    dic = {
        "proc2s": {"FT_mod": 5, "WDW": 2},
        "procs": {"FT_mod": 6, "WDW": 2},
        "acqu2s": {"TD": 144, "SW_h": 1800.0},
        "acqus": acquisition,
    }
    udic = {
        0: {"size": 16, "sw": 900.0, "obs": 70.95, "car": 8283.56, "label": "15N"},
        1: {"size": 256, "sw": 3125.0, "obs": 700.23, "car": 5907.79, "label": "1H"},
    }
    monkeypatch.setattr(spectrum_reader.bruker, "read_pdata", lambda _path: (dic, data))
    monkeypatch.setattr(spectrum_reader.bruker, "guess_udic", lambda *_args, **_kwargs: udic)
    spectra = spectrum_reader.read_spectra(processed)

    assert spectra.params[1].aq_time == pytest.approx(72 / 1800)
    assert spectra.params[2].aq_time == pytest.approx(1994 / 100000)
    # Frequency coordinates still use the extracted width.
    assert spectra.params[2].delta == pytest.approx(-3125 / (256 * 700.23))


def test_bruker_explicit_acquisition_time_and_missing_metadata_fallback():
    explicit = spectrum_reader._bruker_aq_time(
        {"TD": 3988, "SW_h": 100000, "AQ": 0.025}, {}, 3125, direct=True
    )
    fallback = spectrum_reader._bruker_aq_time({"TD": 3988}, {}, 3125, direct=True)
    assert explicit == pytest.approx(0.025)
    assert fallback == pytest.approx(1994 / 3125)
