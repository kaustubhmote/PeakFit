"""Behavior tests for tilted two-dimensional lineshapes."""

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from peakfit.auto_pick.algorithm import _create_peak
from peakfit.engine.algorithms.evaluation import classify_optimizer_result
from peakfit.engine.algorithms.mcmc import estimate_uncertainties_mcmc
from peakfit.engine.algorithms.varpro import fit_cluster
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig, PeakFitConfig
from peakfit.engine.domain.constraints import ParameterConfig, PeakConstraints, apply_constraints
from peakfit.engine.domain.coordinates import rotate_ppm_offsets
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.params_vector import FitParameters
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.state import FittingState
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.lineshapes.gaussian.kernel import kernel as gaussian_kernel
from peakfit.engine.results import FitResult
from peakfit.engine.types import ClusterParameters
from peakfit.fit import fitting
from peakfit.fit.final_outcome import finalize_fit
from peakfit.fit.output_metadata import RunMetadata
from peakfit.fit.pipeline import CorrectionSnapshot, PipelineCompletion
from peakfit.fit.run_models import RunSummary
from peakfit.fit.simulation import FinalModelSnapshot, simulate_final_outcome
from peakfit.io.readers.peaks import read_sparky_list
from peakfit.io.state import load_state, save_state
from peakfit.io.writers.config import WriterConfig
from peakfit.io.writers.orchestrator import write_fit_outputs
from peakfit.plot.reconstruction import SpectraReconstructor
from peakfit.shared.exceptions import DataIOError

if TYPE_CHECKING:
    from peakfit.engine.domain.params_scalar import Parameters

REGISTERED_LINESHAPES = (
    "gaussian",
    "gaussian_doublet",
    "lorentzian",
    "lorentzian_doublet",
    "pvoigt",
    "pvoigt_doublet",
    "sp1",
    "sp1_doublet",
    "sp2",
    "sp2_doublet",
    "no_apod",
    "no_apod_doublet",
)


def test_tilted_fitting_is_opt_in() -> None:
    assert FitConfig().tilted is False
    assert FitConfig(tilted=True).tilted is True


def test_peak_reader_marks_peaks_as_tilted(tmp_path: Path) -> None:
    peak_list = tmp_path / "peaks.list"
    peak_list.write_text("P1 120.0 8.0\n")

    peaks = read_sparky_list(
        peak_list,
        make_2d_spectra(),
        ["gaussian", "gaussian"],
        FitConfig(tilted=True),
    )

    assert len(peaks) == 1
    assert peaks[0].tilted is True


def test_automatic_peak_creation_marks_peaks_as_tilted() -> None:
    peak = _create_peak(
        (4, 5),
        make_2d_spectra(),
        PeakFitConfig(fitting=FitConfig(tilted=True)),
        ["gaussian", "gaussian"],
        "ap1",
    )

    assert peak.tilted is True


def test_per_peak_constraints_accept_a_tilt_axis_pair() -> None:
    constraints = PeakConstraints.model_validate(
        {"F2-F3.tilt": {"value": 35.0, "min": -60.0, "max": 60.0}}
    )

    tilt = constraints.parameters["F2-F3.tilt"]
    assert tilt.value == 35.0
    assert tilt.min == -60.0
    assert tilt.max == 60.0

    spectra = make_2d_spectra()
    peak = Peak(
        name="P1",
        positions=np.array([120.0, 8.0]),
        shapes=create_shapes(
            spectra,
            FitConfig(),
            "P1",
            [120.0, 8.0],
            ["gaussian", "gaussian"],
        ),
        tilted=True,
    )
    params = apply_constraints(
        peak.create_params(),
        ParameterConfig(peaks={"P1": constraints}),
    )
    assert params["P1.F2-F3.tilt"].value == 35.0


def test_loading_rejects_tilted_fit_without_two_spectral_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fitting, "read_spectra", lambda *_args: make_1d_spectra())

    with pytest.raises(
        DataIOError,
        match="Tilted fitting requires exactly two spectral dimensions; input has 1",
    ):
        fitting.load_data(
            Path("ignored.ft1"),
            None,
            None,
            PeakFitConfig(fitting=FitConfig(tilted=True)),
        )


def make_1d_spectra(size: int = 32) -> Spectra:
    return Spectra(
        dic={},
        data=np.zeros((1, size), dtype=np.float64),
        z_values=np.array([], dtype=np.float64),
        params=[
            SpectralParameters(
                size=1,
                sw=1.0,
                obs=1.0,
                car=0.0,
                aq_time=0.1,
                apocode=0.0,
                apodq1=0.0,
                apodq2=0.0,
                apodq3=0.0,
                p180=False,
                direct=False,
                ft=True,
                label="F1",
            ),
            SpectralParameters(
                size=size,
                sw=1200.0,
                obs=600.0,
                car=4.7,
                aq_time=0.25,
                apocode=1.0,
                apodq1=0.2,
                apodq2=0.35,
                apodq3=1.0,
                p180=False,
                direct=True,
                ft=True,
                label="F2",
            ),
        ],
    )


def make_2d_spectra(y_size: int = 9, x_size: int = 11) -> Spectra:
    pseudo = SpectralParameters(
        size=1,
        sw=1.0,
        obs=1.0,
        car=0.0,
        aq_time=0.1,
        apocode=0.0,
        apodq1=0.0,
        apodq2=0.0,
        apodq3=0.0,
        p180=False,
        direct=False,
        ft=True,
        label="F1",
    )
    f2 = SpectralParameters(
        size=y_size,
        sw=90.0,
        obs=100.0,
        car=12000.0,
        aq_time=0.02,
        apocode=1.0,
        apodq1=0.2,
        apodq2=0.35,
        apodq3=1.0,
        p180=False,
        direct=False,
        ft=True,
        label="F2",
    )
    f3 = SpectralParameters(
        size=x_size,
        sw=220.0,
        obs=500.0,
        car=4000.0,
        aq_time=0.01,
        apocode=2.0,
        apodq1=0.1,
        apodq2=0.4,
        apodq3=2.0,
        p180=False,
        direct=True,
        ft=True,
        label="F3",
    )
    return Spectra(
        dic={},
        data=np.zeros((1, y_size, x_size), dtype=np.float64),
        z_values=np.array([0.0]),
        params=[pseudo, f2, f3],
    )


def make_tilted_gaussian_model(
    *,
    tilt: float = 30.0,
    linewidth: float = 25.0,
) -> tuple[Spectra, Peak, Parameters, Cluster]:
    spectra = make_2d_spectra()
    positions = np.array([120.0, 8.0])
    peak = Peak(
        name="P1",
        positions=positions,
        shapes=create_shapes(
            spectra,
            FitConfig(),
            "P1",
            positions.tolist(),
            ["gaussian", "gaussian"],
        ),
        tilted=True,
    )
    peak.set_cluster_id(1)
    params = peak.create_params()
    params["P1.F2-F3.tilt"].value = tilt
    params["P1.F2.lw"].value = linewidth
    params["P1.F3.lw"].value = linewidth
    yy, xx = np.indices((9, 11))
    grid_indices = [yy.ravel(), xx.ravel()]
    cluster = Cluster(1, [peak], grid_indices, np.zeros((99, 1)))
    return spectra, peak, params, cluster


def test_rotate_ppm_offsets_converts_each_principal_axis_to_hz() -> None:
    x_ppm = np.array([[1.0], [0.0]])
    y_ppm = np.array([[0.0], [1.0]])

    u_hz, v_hz = rotate_ppm_offsets(
        x_ppm,
        y_ppm,
        tilt_degrees=np.array([0.0]),
        obs_x_mhz=100.0,
        obs_y_mhz=500.0,
    )

    np.testing.assert_allclose(u_hz, [[100.0], [0.0]])
    np.testing.assert_allclose(v_hz, [[0.0], [500.0]])


def test_rotate_ppm_offsets_uses_counterclockwise_angles_in_ppm_space() -> None:
    root_half = np.sqrt(0.5)
    x_ppm = np.array([[1.0]])
    y_ppm = np.array([[0.0]])

    positive_u, positive_v = rotate_ppm_offsets(
        x_ppm,
        y_ppm,
        tilt_degrees=np.array([45.0]),
        obs_x_mhz=100.0,
        obs_y_mhz=500.0,
    )
    negative_u, negative_v = rotate_ppm_offsets(
        x_ppm,
        y_ppm,
        tilt_degrees=np.array([-45.0]),
        obs_x_mhz=100.0,
        obs_y_mhz=500.0,
    )

    np.testing.assert_allclose(positive_u, [[100.0 * root_half]])
    np.testing.assert_allclose(positive_v, [[-500.0 * root_half]])
    np.testing.assert_allclose(negative_u, positive_u)
    np.testing.assert_allclose(negative_v, -positive_v)


def test_rotated_gaussian_matches_closed_form_reference_value() -> None:
    u_hz, v_hz = rotate_ppm_offsets(
        np.array([[0.04]]),
        np.array([[-0.02]]),
        tilt_degrees=np.array([30.0]),
        obs_x_mhz=100.0,
        obs_y_mhz=500.0,
    )

    value = gaussian_kernel(u_hz, np.array([[10.0]])) * gaussian_kernel(v_hz, np.array([[25.0]]))

    np.testing.assert_allclose(value, [[0.1803180360234895]], rtol=1e-14)


@pytest.mark.parametrize("lineshape", REGISTERED_LINESHAPES)
def test_registered_lineshape_evaluates_precomputed_hz_offsets(lineshape: str) -> None:
    spectra = make_1d_spectra()
    shape = create_shapes(
        spectra,
        FitConfig(fit_phase=["F2"]),
        "P1",
        [4.7],
        [lineshape],
    )[0]
    cluster_params = ClusterParameters(
        extras={
            spec.name: np.array([spec.default], dtype=np.float64)
            for spec in shape.get_parameter_spec()
        }
    )
    grid = np.arange(32, dtype=np.float64)
    offsets_hz, sign = shape.dim_ctx.compute_offsets(
        grid,
        cluster_params.extras["cs"],
    )

    from_grid = shape.evaluate_cluster(grid, cluster_params, compute_derivs=True)
    from_offsets = shape.evaluate_offsets(
        offsets_hz,
        sign,
        cluster_params,
        compute_derivs=True,
    )

    np.testing.assert_allclose(from_offsets.values, from_grid.values)
    offset_derivatives = dict(from_offsets.derivatives)
    offset_derivative = offset_derivatives.pop("offset")
    offset_derivatives["cs"] = offset_derivative * -600.0
    if "dw" in from_grid.derivatives:
        offset_derivatives["dw"] = offset_derivative
    assert offset_derivatives.keys() == from_grid.derivatives.keys()
    for name, expected in from_grid.derivatives.items():
        np.testing.assert_allclose(offset_derivatives[name], expected)


def test_tilted_peak_creates_one_joint_angle_parameter() -> None:
    spectra = make_2d_spectra()
    shapes = create_shapes(
        spectra,
        FitConfig(),
        "P1",
        [120.0, 8.0],
        ["gaussian", "gaussian"],
    )

    params = Peak(
        name="P1",
        positions=np.array([120.0, 8.0]),
        shapes=shapes,
        tilted=True,
    ).create_params()

    tilt = params["P1.F2-F3.tilt"]
    assert tilt.value == 0.0
    assert tilt.min == -89.0
    assert tilt.max == 89.0
    assert tilt.unit == "deg"


def test_cluster_evaluates_closed_form_tilted_gaussian() -> None:
    spectra = make_2d_spectra()
    shapes = create_shapes(
        spectra,
        FitConfig(),
        "P1",
        [120.0, 8.0],
        ["gaussian", "gaussian"],
    )
    peak = Peak(
        name="P1",
        positions=np.array([120.0, 8.0]),
        shapes=shapes,
        tilted=True,
    )
    peak.set_cluster_id(1)
    params = peak.create_params()
    params["P1.F2-F3.tilt"].value = 30.0
    yy, xx = np.indices((9, 11))
    grid_indices = [yy.ravel(), xx.ravel()]
    cluster = Cluster(
        cluster_id=1,
        peaks=[peak],
        grid_indices=grid_indices,
        data=np.zeros((99, 1)),
    )

    actual = cluster.evaluate(params)[0]

    f2_hz, _ = shapes[0].dim_ctx.compute_offsets(
        grid_indices[0].astype(np.float64),
        np.array([120.0]),
    )
    f3_hz, _ = shapes[1].dim_ctx.compute_offsets(
        grid_indices[1].astype(np.float64),
        np.array([8.0]),
    )
    theta = np.deg2rad(30.0)
    u_hz = 100.0 * (f2_hz / 100.0 * np.cos(theta) + f3_hz / 500.0 * np.sin(theta))
    v_hz = 500.0 * (-f2_hz / 100.0 * np.sin(theta) + f3_hz / 500.0 * np.cos(theta))
    expected = np.exp(-4.0 * np.log(2.0) * ((u_hz / 25.0) ** 2 + (v_hz / 25.0) ** 2)).ravel()

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("lineshape", REGISTERED_LINESHAPES)
def test_zero_tilt_matches_axis_aligned_cluster_evaluation(lineshape: str) -> None:
    spectra = make_2d_spectra()
    config = FitConfig(fit_phase=["F2", "F3"])
    positions = np.array([120.0, 8.0])
    axis_peak = Peak(
        name="P1",
        positions=positions,
        shapes=create_shapes(
            spectra,
            config,
            "P1",
            positions.tolist(),
            [lineshape, lineshape],
        ),
    )
    tilted_peak = Peak(
        name="P1",
        positions=positions,
        shapes=create_shapes(
            spectra,
            config,
            "P1",
            positions.tolist(),
            [lineshape, lineshape],
        ),
        tilted=True,
    )
    axis_peak.set_cluster_id(1)
    tilted_peak.set_cluster_id(1)
    yy, xx = np.indices((9, 11))
    grid_indices = [yy.ravel(), xx.ravel()]
    data = np.zeros((99, 1))

    axis_values = Cluster(1, [axis_peak], grid_indices, data).evaluate(axis_peak.create_params())
    tilted_values = Cluster(1, [tilted_peak], grid_indices, data).evaluate(
        tilted_peak.create_params()
    )

    np.testing.assert_allclose(tilted_values, axis_values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("lineshape", REGISTERED_LINESHAPES)
def test_tilted_cluster_analytic_derivatives_match_finite_differences(
    lineshape: str,
) -> None:
    spectra = make_2d_spectra()
    config = FitConfig(fit_phase=["F2", "F3"])
    positions = np.array([120.0, 8.0])
    peak = Peak(
        name="P1",
        positions=positions,
        shapes=create_shapes(
            spectra,
            config,
            "P1",
            positions.tolist(),
            [lineshape, lineshape],
        ),
        tilted=True,
    )
    peak.set_cluster_id(1)
    params = peak.create_params()
    params["P1.F2-F3.tilt"].value = 23.0
    yy, xx = np.indices((7, 9))
    yy += 1
    xx += 1
    grid_indices = [yy.ravel(), xx.ravel()]
    cluster = Cluster(1, [peak], grid_indices, np.zeros((63, 1)))
    names = params.get_vary_names()
    parameter_map = ParameterMap.from_names(names)
    derivatives = np.zeros((len(names), 63), dtype=np.float64)

    cluster.evaluate(
        params,
        d_matrix=derivatives,
        param_map=parameter_map,
    )

    for index, name in enumerate(names):
        original = params[name].value
        step = 1e-5 if params[name].unit != "ppm" else 1e-6
        params[name].value = original + step
        plus = cluster.evaluate(params)[0]
        params[name].value = original - step
        minus = cluster.evaluate(params)[0]
        params[name].value = original
        finite_difference = (plus - minus) / (2.0 * step)

        np.testing.assert_allclose(
            derivatives[index],
            finite_difference,
            rtol=2e-4,
            atol=1e-8,
            err_msg=name,
        )


def test_spectrum_reconstruction_does_not_clip_a_tilted_peak(tmp_path: Path) -> None:
    _spectra, peak, params, cluster = make_tilted_gaussian_model(
        tilt=45.0,
        linewidth=5.0,
    )
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(params, [peak]),
        scalar_params=params,
        noise=1.0,
    )
    reconstructor = object.__new__(SpectraReconstructor)
    reconstructor.results_dir = tmp_path
    reconstructor._state = state
    reconstructor._cluster_all = cluster
    reconstructor._amplitudes = np.array([[2.0]])

    reconstructed = reconstructor.reconstruct_plane(0, (9, 11))
    expected = (2.0 * cluster.evaluate(params)[0]).reshape(9, 11)

    np.testing.assert_allclose(reconstructed, expected)


def test_tilt_survives_finalization_outputs_and_full_grid_simulation(tmp_path: Path) -> None:
    spectra, peak, params, empty_cluster = make_tilted_gaussian_model(tilt=32.0)
    basis = empty_cluster.evaluate(params)
    amplitudes = np.array([[2.0]])
    cluster = Cluster(
        cluster_id=1,
        peaks=[peak],
        grid_indices=empty_cluster.grid_indices,
        data=basis.T @ amplitudes,
    )
    result = FitResult(
        cluster_id=1,
        params=params.copy(deep=True),
        residual=np.zeros(cluster.n_observations),
        cost=0.0,
        nfev=1,
        njev=1,
        success=True,
        message="converged",
        n_amplitude_params=1,
        optimizer_kind="varpro",
        correction_revision=0,
        noise=1.0,
    )
    evaluation = classify_optimizer_result(cluster=cluster, result=result, noise=1.0)
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(params, [peak]),
        scalar_params=params,
        noise=1.0,
    )
    state_path = save_state(tmp_path / "metadata" / "fitting_state.pkl", state)
    restored_state = load_state(state_path)
    assert restored_state.peaks[0].tilted is True
    assert "P1.F2-F3.tilt" in restored_state.scalar_params

    correction_snapshot = CorrectionSnapshot.from_clusters([cluster], revision=0)
    completion = PipelineCompletion(
        state=state,
        results=[result],
        evaluations=[evaluation],
        correction_snapshot=correction_snapshot,
        n_optimizer_passes=1,
    )
    outcome = finalize_fit(completion)

    tilt_names = {parameter.name for parameter in outcome.final_nonlinear_parameters}
    assert "P1.F2-F3.tilt" in tilt_names

    snapshot = FinalModelSnapshot.capture(outcome, state, correction_snapshot)
    simulated = simulate_final_outcome(outcome, snapshot, spectra.data)
    assert simulated is not None
    np.testing.assert_allclose(simulated, cluster.data.T.reshape(1, 9, 11))

    written = write_fit_outputs(
        outcome,
        tmp_path,
        WriterConfig(formats=("json", "csv")),
        metadata=RunMetadata(),
        z_values=spectra.z_values,
        summary=RunSummary.from_outcome(outcome),
    )
    payload = json.loads(written["summary_fit"].read_text())
    json_names = {parameter["name"] for parameter in payload["final_nonlinear_parameters"]}
    assert "P1.F2-F3.tilt" in json_names
    reconstructor = SpectraReconstructor(tmp_path)
    reconstructed = reconstructor.reconstruct_plane(0, (9, 11))
    np.testing.assert_allclose(reconstructed, simulated[0])
    y_axis, x_axis = spectra.spectral_params
    xlim = (float(x_axis.pts2ppm(0)), float(x_axis.pts2ppm(10)))
    ylim = (float(y_axis.pts2ppm(0)), float(y_axis.pts2ppm(8)))
    reconstructor.validate_spectrum(spectra.data.shape, xlim, ylim)
    with pytest.raises(ValueError, match="different plane counts"):
        reconstructor.validate_spectrum((2, 9, 11), xlim, ylim)
    with pytest.raises(ValueError, match="different ppm grids"):
        reconstructor.validate_spectrum(spectra.data.shape, (xlim[0] + 0.1, xlim[1] + 0.1), ylim)
    # Persisted observations are not a second source of fitted amplitudes.
    state.clusters[0].data *= 7
    save_state(state_path, state)
    np.testing.assert_allclose(
        SpectraReconstructor(tmp_path).reconstruct_plane(0, (9, 11)), simulated[0]
    )
    with written["parameters"].open(newline="") as handle:
        csv_names = {row["parameter_name"] for row in csv.DictReader(handle)}
    assert "P1.F2-F3.tilt" in csv_names


def test_mcmc_samples_tilt_as_a_nonlinear_parameter() -> None:
    _spectra, _peak, params, cluster = make_tilted_gaussian_model(tilt=20.0)
    for parameter in params.values():
        parameter.vary = parameter.name == "P1.F2-F3.tilt"

    result = estimate_uncertainties_mcmc(
        params,
        cluster,
        noise=1.0,
        n_walkers=4,
        n_steps=4,
        burn_in=0,
        workers=1,
    )

    assert result.n_lineshape_params == 1
    assert result.parameter_names[0] == "P1.F2-F3.tilt"


@pytest.mark.parametrize("lineshape", ["gaussian", "pvoigt", "no_apod"])
def test_varpro_recovers_tilt_from_synthetic_peak(lineshape: str) -> None:
    spectra = make_2d_spectra()
    positions = np.array([120.0, 8.0])
    peak = Peak(
        name="P1",
        positions=positions,
        shapes=create_shapes(
            spectra,
            FitConfig(),
            "P1",
            positions.tolist(),
            [lineshape, lineshape],
        ),
        tilted=True,
    )
    peak.set_cluster_id(1)
    true_params = peak.create_params()
    true_params["P1.F2-F3.tilt"].value = 28.0
    true_params["P1.F2.lw"].value = 12.0
    true_params["P1.F3.lw"].value = 30.0
    yy, xx = np.indices((7, 9))
    yy += 1
    xx += 1
    grid_indices = [yy.ravel(), xx.ravel()]
    empty_cluster = Cluster(1, [peak], grid_indices, np.zeros((63, 1)))
    data = empty_cluster.evaluate(true_params).T @ np.array([[100.0]])
    cluster = Cluster(1, [peak], grid_indices, data)
    fit_params = true_params.copy(deep=True)
    for parameter in fit_params.values():
        parameter.vary = parameter.name == "P1.F2-F3.tilt"
    fit_params["P1.F2-F3.tilt"].value = 10.0

    result = fit_cluster(
        fit_params,
        cluster,
        noise=1.0,
        max_nfev=200,
    )

    assert result.success
    assert result.params["P1.F2-F3.tilt"].value == pytest.approx(28.0, abs=0.02)


def test_spectrum_reconstruction_retains_lorentzian_tails(tmp_path: Path) -> None:
    spectra = make_2d_spectra(101, 101)
    peak = Peak(
        name="P1",
        positions=np.array([120.0, 8.0]),
        shapes=create_shapes(spectra, FitConfig(), "P1", [120.0, 8.0], ["lorentzian"] * 2),
    )
    peak.set_cluster_id(1)
    params = peak.create_params()
    params["P1.F2.lw"].value = 0.2
    params["P1.F3.lw"].value = 0.2
    yy, xx = np.indices((101, 101))
    cluster = Cluster(1, [peak], [yy.ravel(), xx.ravel()], np.zeros((101 * 101, 1)))
    state = FittingState(
        clusters=[cluster],
        params=FitParameters.from_parameters(params, [peak]),
        scalar_params=params,
    )
    reconstructor = object.__new__(SpectraReconstructor)
    reconstructor.results_dir = tmp_path
    reconstructor._state = state
    reconstructor._cluster_all = cluster
    reconstructor._amplitudes = np.array([[2.0]])
    actual = reconstructor.reconstruct_plane(0, (101, 101))
    expected = (2 * cluster.evaluate(params)[0]).reshape(101, 101)
    assert expected[0, 0] > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)
