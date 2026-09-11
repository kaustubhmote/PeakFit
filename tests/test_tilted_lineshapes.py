"""Behavior tests for tilted two-dimensional lineshapes."""

import numpy as np

from peakfit.engine.lineshapes.gaussian.kernel import kernel as gaussian_kernel
from peakfit.engine.lineshapes.tilt import rotate_ppm_offsets


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

    value = gaussian_kernel(u_hz, np.array([[10.0]])) * gaussian_kernel(
        v_hz, np.array([[25.0]])
    )

    np.testing.assert_allclose(value, [[0.1803180360234895]], rtol=1e-14)
