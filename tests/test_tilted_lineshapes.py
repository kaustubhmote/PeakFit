"""Behavior tests for tilted two-dimensional lineshapes."""

import numpy as np
import pytest

from peakfit.engine.domain.config import FitConfig
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
