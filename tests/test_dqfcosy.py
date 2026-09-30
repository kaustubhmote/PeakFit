"""Scientific and workflow protection for four-lobe DQF-COSY fitting."""

import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from peakfit.engine.algorithms.clustering import create_clusters
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.config import DQFCOSYWindow, FitConfig, PeakFitConfig
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, get_shape_names
from peakfit.engine.domain.state import FittingState
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.lineshapes.dqfcosy import DQFCOSY, antiphase_profile, windowed_resonance
from peakfit.fit import fitting
from peakfit.fit.output_metadata import RunMetadata
from peakfit.fit.simulation import simulate_final_outcome
from peakfit.io.readers import spectrum as spectrum_reader
from peakfit.io.state import load_state, save_state
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import write_fit_outputs
from peakfit.shared.exceptions import DataIOError


def make_spectra() -> Spectra:
    params = [
        SpectralParameters(
            size=161,
            sw=80.5,
            obs=600.0,
            car=center * 600,
            aq_time=0.25,
            apocode=1,
            apodq1=0.15,
            apodq2=1,
            apodq3=1,
            p180=False,
            direct=i == 2,
            ft=True,
            label=f"F{i + 1}",
            nucleus="1H",
        )
        for i, center in enumerate([0, 8.0, 4.5])
    ]
    return Spectra(dic={}, params=params, data=np.zeros((1, 161, 161)), z_values=np.array([0.0]))


def make_peak(spectra: Spectra, config: FitConfig) -> Peak:
    return Peak(
        "A23",
        np.array([8.0, 4.5]),
        create_shapes(spectra, config, "A23", [8.0, 4.5], get_shape_names(config, spectra)),
    )


def direct_transform(offsets, lw, coupling, window):
    """Independent sampled FID, including signed J evolution, transformed by summation."""
    t = np.arange(window.points) * window.acquisition_time / window.points
    weights = np.exp(-np.pi * (lw + window.lb_hz) * t)
    if window.sine_power:
        weights *= (
            np.sin(np.linspace(np.pi * window.sine_offset, np.pi * window.sine_end, window.points))
            ** window.sine_power
        )
    fid = 2j * np.sin(np.pi * coupling * t) * weights
    return np.exp(-2j * np.pi * np.outer(offsets, t)) @ fid / window.points


@pytest.mark.parametrize("power", [0, 1, 2])
@pytest.mark.parametrize("lw", [0.0, 0.1, 6.0, 30.0])
def test_matches_independent_sampled_fid(power, lw):
    window = DQFCOSYWindow(
        acquisition_time=0.064, points=64, sine_power=power, sine_offset=0.12, lb_hz=0.0
    )
    offsets = np.array([-1000, -100, -9, 0, 9, 100, 1000.0])
    actual, derivatives = antiphase_profile(offsets, np.array(lw), np.array(5.0), window)
    np.testing.assert_allclose(actual, direct_transform(offsets, lw, 5, window), atol=2e-13)
    assert all(np.isfinite(d).all() for d in derivatives.values())


@pytest.mark.parametrize("power", [0, 1, 2])
def test_analytic_derivatives(power):
    window = DQFCOSYWindow(
        acquisition_time=0.3, points=512, sine_power=power, sine_offset=0.1, lb_hz=1.5
    )
    offsets = np.linspace(-25, 25, 151)[:, None]
    lw, j = np.array([4.5, 11.0]), np.array([4.2, 9.0])
    _, derivs = antiphase_profile(offsets, lw, j, window)
    step = 1e-4
    for key, args in {
        "dw": (offsets, lw, j),
        "lw": (offsets, lw, j),
        "j": (offsets, lw, j),
    }.items():
        index = {"dw": 0, "lw": 1, "j": 2}[key]
        plus, minus = list(args), list(args)
        plus[index] = args[index] + step
        minus[index] = args[index] - step
        expected = (
            antiphase_profile(plus[0], plus[1], plus[2], window)[0]
            - antiphase_profile(minus[0], minus[1], minus[2], window)[0]
        ) / (2 * step)
        np.testing.assert_allclose(derivs[key], expected, rtol=2e-5, atol=2e-10)


def test_relaxation_cancellation_and_lobe_separation():
    window = DQFCOSYWindow(acquisition_time=1, points=20000)
    offsets = np.linspace(-25, 25, 10001)
    value, _ = antiphase_profile(offsets, np.array(10.0), np.array(5.0), window)
    np.testing.assert_allclose(value.real, -value.real[::-1], atol=1e-13)
    assert value.real[len(offsets) // 2] == pytest.approx(0, abs=1e-14)
    separation = offsets[np.argmax(value.real)] - offsets[np.argmin(value.real)]
    assert separation > 6.0  # Maxima separation overestimates the actual 5-Hz J.
    single, _ = windowed_resonance(np.array([0.0]), np.array(10.0), window)
    assert np.max(value.real) < single.real[0]


def test_shared_j_cluster_jacobian_and_center_segmentation():
    spectra = make_spectra()
    config = FitConfig(lineshape="dqfcosy", fit_phase=["F2", "F3"], dqfcosy_region_hz=22)
    peak = make_peak(spectra, config)
    # This must produce a useful fitting rectangle even with zero signal at its center.
    clusters = create_clusters(spectra, [peak], contour_level=1)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.n_points > 7000
    params = peak.create_params()
    assert [name for name in params if name.endswith(".j")] == ["A23.F2-F3.j"]
    params["A23.F2-F3.j"].value = 5.1
    np.testing.assert_array_equal(
        FitParameters.from_parameters(params, [peak]).extras["j"], [5.1, 5.1]
    )
    names = params.get_vary_names()
    mapping = ParameterMap.from_names(names)
    jac = np.zeros((len(names), cluster.n_points))
    cluster.evaluate(params, d_matrix=jac, param_map=mapping)
    for i, name in enumerate(names):
        original = params[name].value
        step = 1e-6 if name.endswith(".cs") else 1e-4
        params[name].value = original + step
        plus = cluster.evaluate(params)
        params[name].value = original - step
        minus = cluster.evaluate(params)
        params[name].value = original
        np.testing.assert_allclose(jac[i], ((plus - minus) / (2 * step))[0], rtol=3e-4, atol=1e-8)


@pytest.mark.parametrize("power", [0, 1, 2])
def test_recovers_j_from_independent_noisy_2d_data(power):
    spectra = make_spectra()
    windows = {
        "F2": DQFCOSYWindow(
            acquisition_time=0.08, points=128, sine_power=power, sine_offset=0.1, lb_hz=0.5
        ),
        "F3": DQFCOSYWindow(
            acquisition_time=0.25, points=512, sine_power=power, sine_offset=0.0, lb_hz=1.0
        ),
    }
    config = FitConfig(lineshape="dqfcosy", dqfcosy_windows=windows, dqfcosy_region_hz=30)
    peak = make_peak(spectra, config)
    # True centers differ from the supplied midpoint guesses.
    positions = [8.001, 4.498]
    true_j = 4.2
    profiles = [
        direct_transform(
            spec.pts2hz(np.arange(spec.size)) - pos * spec.obs, lw, true_j, windows[spec.label]
        ).real
        for spec, pos, lw in zip(spectra.spectral_params, positions, [6.0, 5.0], strict=True)
    ]
    data = np.outer(*profiles)
    data /= np.max(np.abs(data))
    noise = 0.001
    spectra.data[0] = data + np.random.default_rng(18).normal(0, noise, data.shape)
    cluster = create_clusters(spectra, [peak], contour_level=0.1)[0]
    result = fit_cluster(peak.create_params(), cluster, noise=noise, max_nfev=200)
    assert result.success
    assert result.params["A23.F2-F3.j"].value == pytest.approx(true_j, abs=0.08)
    assert result.params["A23.F2.lw"].value == pytest.approx(6.0, abs=0.1)
    assert result.params["A23.F3.lw"].value == pytest.approx(5.0, abs=0.1)
    assert result.params["A23.F2.cs"].value == pytest.approx(positions[0], abs=2e-5)
    assert result.params["A23.F3.cs"].value == pytest.approx(positions[1], abs=2e-5)


def test_window_override_and_unsupported_headers():
    spectra = make_spectra()
    spectra.params[1].apocode = -1
    with pytest.raises(ValueError, match="unsupported or unknown window"):
        make_peak(spectra, FitConfig(lineshape="dqfcosy"))
    override = DQFCOSYWindow(acquisition_time=0.1, points=256)
    peak = make_peak(spectra, FitConfig(lineshape="dqfcosy", dqfcosy_windows={"F2": override}))
    assert isinstance(peak.shapes[0], DQFCOSY)
    assert peak.shapes[0].window == override
    with pytest.raises(ValueError, match="Unknown DQF-COSY window axes"):
        make_peak(spectra, FitConfig(lineshape="dqfcosy", dqfcosy_windows={"F1": override}))
    spectra.dic["_peakfit_format"] = "bruker"
    with pytest.raises(ValueError, match="Bruker input"):
        make_peak(spectra, FitConfig(lineshape="dqfcosy"))


def test_extracted_axis_requires_original_acquisition_timing():
    spectra = make_spectra()
    spectra.params[2].ft_size = 1024
    with pytest.raises(ValueError, match="original acquisition timing"):
        make_peak(spectra, FitConfig(lineshape="dqfcosy"))
    window = DQFCOSYWindow(acquisition_time=0.2, points=512)
    peak = make_peak(spectra, FitConfig(lineshape="dqfcosy", dqfcosy_windows={"F3": window}))
    assert isinstance(peak.shapes[1], DQFCOSY)
    assert peak.shapes[1].window == window


@pytest.mark.parametrize(
    "options",
    [
        {"points": 1},
        {"acquisition_time": 0},
        {"sine_power": 3},
        {"lb_hz": -2},
        {"sine_offset": float("nan")},
    ],
)
def test_invalid_window_inputs(options):
    with pytest.raises(ValidationError):
        DQFCOSYWindow.model_validate({"points": 256, "acquisition_time": 0.2, **options})


def test_dqfcosy_requires_center_list_and_rejects_tilt(monkeypatch):
    monkeypatch.setattr(fitting, "read_spectra", lambda *_args: make_spectra())
    with pytest.raises(DataIOError, match="peak list marking the centers"):
        fitting.load_data(
            Path("unused"), None, None, PeakFitConfig(fitting=FitConfig(lineshape="dqfcosy"))
        )
    with pytest.raises(ValueError, match="tilted=false"):
        make_peak(make_spectra(), FitConfig(lineshape="dqfcosy", tilted=True))


def test_full_pipeline_exports_and_restores_shared_j(tmp_path, monkeypatch):
    spectra = make_spectra()
    window = DQFCOSYWindow(acquisition_time=0.2, points=512, sine_power=2, sine_offset=0.2)
    config = PeakFitConfig(
        fitting=FitConfig(
            lineshape="dqfcosy",
            dqfcosy_windows={"F2": window, "F3": window},
        ),
        noise_level=1e-4,
    )
    profiles = [
        direct_transform(
            spec.pts2hz(np.arange(spec.size)) - center * spec.obs, 5.0, 6.2, window
        ).real
        for spec, center in zip(spectra.spectral_params, [8.0, 4.5], strict=True)
    ]
    spectra.data[0] = np.outer(*profiles)
    spectra.data /= np.max(np.abs(spectra.data))
    monkeypatch.setattr(fitting, "read_spectra", lambda *_args: spectra)
    peak_list = tmp_path / "centers.list"
    peak_list.write_text("A23 8.0 4.5\n")
    data = fitting.load_data(tmp_path / "spectrum.ft2", peak_list, None, config)
    run = fitting.run_fit(data, config, tmp_path / "results", workers=1)
    assert run.success
    parameters = {p.name: p for p in run.outcome.final_nonlinear_parameters}
    assert parameters["A23.F2-F3.j"].value == pytest.approx(6.2, abs=1e-5)
    assert parameters["A23.F2-F3.j"].unit == "Hz"
    written = write_fit_outputs(
        run.outcome,
        run.output_dir,
        WriterConfig(formats=("csv", "json")),
        metadata=RunMetadata(),
        z_values=spectra.z_values,
        summary=run.summary,
    )
    with written["parameters"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    j_rows = [row for row in rows if row["parameter_name"].endswith(".j")]
    assert len(j_rows) == 1
    assert float(j_rows[0]["value"]) == pytest.approx(6.2, abs=1e-5)
    payload = json.loads(written["summary_fit"].read_text())
    assert "A23.F2-F3.j" in {p["name"] for p in payload["final_nonlinear_parameters"]}
    restored = load_state(save_state(tmp_path / "state.pkl", run.continuation_state))
    np.testing.assert_allclose(restored.params.extras["j"], [6.2, 6.2], atol=1e-5)
    np.testing.assert_allclose(
        restored.clusters[0].evaluate(restored.scalar_params),
        run.continuation_state.clusters[0].evaluate(run.continuation_state.scalar_params),
    )
    assert run.simulation_snapshot is not None
    reconstructed = simulate_final_outcome(run.outcome, run.simulation_snapshot, spectra.data)
    assert reconstructed is not None
    np.testing.assert_allclose(reconstructed, spectra.data, atol=1e-6)


def test_bruker_header_defaults_recover_j_without_window_configuration(tmp_path, monkeypatch):
    # Acquisition widths differ from the cropped displayed axes. Both are zero filled.
    source = make_spectra()
    dictionaries = {
        "acqu2s": {"TD": 256, "SW_h": 1600.0, "FnMODE": 5},
        "acqus": {"TD": 1024, "SW_h": 2048.0, "AQ_mod": 3},
        "proc2s": {"WDW": 3, "SSB": 3, "LB": 100.0, "FT_mod": 5, "FTSIZE": 8192, "SI": 8192},
        "procs": {"WDW": 4, "SSB": 0, "FT_mod": 6, "FTSIZE": 4096, "SI": 4096},
    }
    profiles = []
    centers = [8.001, 4.498]
    for spec, center, width, sw, points, power, offset in zip(
        source.spectral_params,
        centers,
        [6, 5],
        [1600, 2048],
        [128, 512],
        [1, 2],
        [1 / 3, 0],
        strict=True,
    ):
        t = np.arange(points) / sw
        window = np.sin(np.pi * (offset + (1 - offset) * t / (points / sw))) ** power
        fid = 2j * np.sin(np.pi * 4.2 * t) * np.exp(-np.pi * width * t) * window
        offsets = spec.pts2hz(np.arange(spec.size)) - center * spec.obs
        profiles.append((np.exp(-2j * np.pi * np.outer(offsets, t)) @ fid / points).real)
    data = np.outer(*profiles)
    data /= np.max(np.abs(data))
    data += np.random.default_rng(18).normal(0, 0.001, data.shape)
    udic = {
        i: {"size": spec.size, "sw": spec.sw, "obs": spec.obs, "car": spec.car, "label": "1H"}
        for i, spec in enumerate(source.spectral_params)
    }
    processed = tmp_path / "pdata" / "1"
    processed.mkdir(parents=True)
    (processed / "procs").touch()
    (processed / "2rr").touch()
    peak_list = tmp_path / "centers.list"
    peak_list.write_text("A23 8.0 4.5\n")
    monkeypatch.setattr(spectrum_reader.bruker, "read_pdata", lambda _path: (dictionaries, data))
    monkeypatch.setattr(spectrum_reader.bruker, "guess_udic", lambda *_args, **_kwargs: udic)
    config = PeakFitConfig(fitting=FitConfig(lineshape="dqfcosy"), noise_level=0.001)
    loaded = fitting.load_data(processed, peak_list, None, config)
    run = fitting.run_fit(loaded, config, tmp_path / "results", workers=1)
    assert run.success
    j = next(p for p in run.outcome.final_nonlinear_parameters if p.name == "A23.F2-F3.j")
    assert j.value == pytest.approx(4.2, abs=0.08)
    assert j.unit == "Hz"


def quadrature_example(components, *, noise=0.0, phases=(3.0, -5.0)):
    """Build each component independently from sampled complex time signals."""
    spectra = make_spectra()
    windows = {
        "F2": DQFCOSYWindow(acquisition_time=0.046, points=192, sine_power=1),
        "F3": DQFCOSYWindow(acquisition_time=0.30, points=512, sine_power=2, sine_offset=0.2),
    }
    config = FitConfig(
        lineshape="dqfcosy",
        use_imaginary=True,
        dqfcosy_windows=windows,
        fit_phase=["F2", "F3"],
        dqfcosy_region_hz=28,
    )
    profiles = [
        direct_transform(
            spec.pts2hz(np.arange(spec.size)) - center * spec.obs, width, 4.2, windows[spec.label]
        )
        * np.exp(1j * np.deg2rad(phase))
        for spec, center, width, phase in zip(
            spectra.spectral_params, [8.0, 4.5], [12.0, 6.0], phases, strict=True
        )
    ]
    planes = np.stack(
        [
            np.outer(
                -profiles[0].imag if code[1] == "i" else profiles[0].real,
                -profiles[1].imag if code[0] == "i" else profiles[1].real,
            )
            for code in components
        ]
    )
    planes /= np.max(np.abs(planes))
    planes += np.random.default_rng(34).normal(0, noise, planes.shape)
    spectra = Spectra(
        dic=spectra.dic,
        params=spectra.params,
        data=planes,
        z_values=np.arange(len(components), dtype=float),
        quadrature_components=components,
    )
    return spectra, config


@pytest.mark.parametrize("components", [("rr", "ri"), ("rr", "ir", "ii"), ("rr", "ri", "ir", "ii")])
def test_joint_quadrature_recovers_j_width_and_phase(components):
    spectra, config = quadrature_example(components, noise=0.002)
    peak = make_peak(spectra, config)
    cluster = create_clusters(spectra, [peak], contour_level=2)[0]
    assert cluster.n_series == 1
    assert cluster.n_observations % len(components) == 0
    result = fit_cluster(peak.create_params(), cluster, noise=0.002, max_nfev=200)
    assert result.success
    assert result.params["A23.F2-F3.j"].value == pytest.approx(4.2, abs=0.08)
    assert result.params["A23.F2.lw"].value == pytest.approx(12.0, abs=0.15)
    assert result.params["A23.F3.lw"].value == pytest.approx(6.0, abs=0.1)
    for shape, expected in zip(peak.shapes, [3.0, -5.0], strict=True):
        assert result.params[shape.parameter_name("phase")].value == pytest.approx(
            expected, abs=0.15
        )
    assert [name for name in result.params if name.endswith(".j")] == ["A23.F2-F3.j"]


def test_joint_quadrature_jacobian_includes_both_shared_j_derivatives():
    spectra, config = quadrature_example(("rr", "ri", "ir", "ii"))
    peak = make_peak(spectra, config)
    cluster = create_clusters(spectra, [peak], contour_level=2)[0]
    params = peak.create_params()
    names = params.get_vary_names()
    jacobian = np.zeros((len(names), cluster.n_points))
    cluster.evaluate(params, d_matrix=jacobian, param_map=ParameterMap.from_names(names))
    for i, name in enumerate(names):
        original = params[name].value
        step = 1e-6 if name.endswith(".cs") else 1e-4
        params[name].value = original + step
        plus = cluster.evaluate(params)
        params[name].value = original - step
        minus = cluster.evaluate(params)
        params[name].value = original
        np.testing.assert_allclose(
            jacobian[i], ((plus - minus) / (2 * step))[0], rtol=3e-4, atol=1e-9
        )


def test_joint_quadrature_pipeline_state_output_and_simulation(tmp_path, monkeypatch):
    spectra, fit_config = quadrature_example(("rr", "ri", "ir", "ii"), phases=(0.0, 0.0))
    config = PeakFitConfig(fitting=fit_config, noise_level=0.001)
    config.output.include_timestamp = False
    config.output.save_simulated = True
    monkeypatch.setattr(fitting, "read_spectra", lambda *_args, **_kwargs: spectra)
    peaklist = tmp_path / "centers.list"
    peaklist.write_text("A23 8.0 4.5\n")
    data = fitting.load_data(tmp_path / "real.ft2", peaklist, None, config)
    run = fitting.run_fit(data, config, tmp_path / "results", workers=1)
    assert run.success
    fitting.write_fit_run_outputs(run, spectra, config, {"peaklist": peaklist})
    document = json.loads((run.output_dir / "summary/fit.json").read_text())
    assert document["metadata"]["quadrature_components"] == ["rr", "ri", "ir", "ii"]
    assert document["z_axis"]["values"] == [0.0]
    assert (
        document["clusters"][0]["analytical_evaluation"]["statistics"]["n_amplitude_parameters"]
        == 1
    )
    restored = load_state(run.output_dir / "metadata/fitting_state.pkl")
    assert restored.clusters[0].quadrature_components == spectra.quadrature_components
    np.testing.assert_allclose(
        restored.clusters[0].evaluate(restored.scalar_params),
        run.continuation_state.clusters[0].evaluate(run.continuation_state.scalar_params),
    )
    assert run.simulation_snapshot is not None
    simulated = simulate_final_outcome(run.outcome, run.simulation_snapshot, spectra.data)
    assert simulated is not None
    np.testing.assert_allclose(simulated, spectra.data, atol=1e-5)
    with np.load(run.output_dir / "simulated_quadrature.npz") as saved:
        assert set(saved.files) == set(spectra.quadrature_components)
        np.testing.assert_allclose(saved["ri"], spectra.data[1], atol=1e-5)


def test_legacy_real_only_state_still_evaluates(tmp_path):
    spectra = make_spectra()
    peak = make_peak(spectra, FitConfig(lineshape="dqfcosy"))
    cluster = create_clusters(spectra, [peak], contour_level=2)[0]
    params = peak.create_params()
    expected = cluster.evaluate(params)
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(params, [peak]),
        scalar_params=params,
        noise=0.001,
    )
    payload = state.model_dump()
    payload["clusters"][0].pop("quadrature_components")
    payload["clusters"][0].pop("component_indices")
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(payload))
    restored = load_state(path)
    assert restored.clusters[0].quadrature_components == ()
    np.testing.assert_allclose(restored.clusters[0].evaluate(restored.scalar_params), expected)


def test_joint_quadrature_reduces_j_linewidth_ambiguity():
    spectra, config = quadrature_example(("rr", "ri", "ir", "ii"), noise=0.002)
    peak = make_peak(spectra, config)
    joint = create_clusters(spectra, [peak], contour_level=2)[0]
    result_joint = fit_cluster(peak.create_params(), joint, noise=0.002)
    real = spectra.model_copy(update={"data": spectra.data[:1], "quadrature_components": ()})
    real_peak = make_peak(real, config)
    cluster_real = create_clusters(real, [real_peak], contour_level=2)[0]
    result_real = fit_cluster(real_peak.create_params(), cluster_real, noise=0.002)
    assert result_joint.success
    assert result_real.success
    error_joint = result_joint.params["A23.F2-F3.j"].stderr
    error_real = result_real.params["A23.F2-F3.j"].stderr
    assert error_joint is not None
    assert error_real is not None
    assert error_joint < error_real * 0.7
