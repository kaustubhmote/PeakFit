"""Behavior tests for tilted two-dimensional lineshapes."""

import numpy as np
import pytest

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.engine.lineshapes.gaussian.kernel import kernel as gaussian_kernel
from peakfit.engine.lineshapes.tilt import rotate_ppm_offsets
from peakfit.engine.types import ClusterParameters

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
