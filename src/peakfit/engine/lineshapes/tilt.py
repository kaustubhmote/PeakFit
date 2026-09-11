"""Coordinate rotation for tilted two-dimensional lineshapes."""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray


def rotate_ppm_offsets(
    x_ppm: npt.ArrayLike,
    y_ppm: npt.ArrayLike,
    *,
    tilt_degrees: npt.ArrayLike,
    obs_x_mhz: float,
    obs_y_mhz: float,
) -> tuple[FloatArray, FloatArray]:
    """Rotate ppm offsets and convert the principal coordinates to Hz."""
    x = np.asarray(x_ppm, dtype=np.float64)
    y = np.asarray(y_ppm, dtype=np.float64)
    theta = np.deg2rad(np.asarray(tilt_degrees, dtype=np.float64))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    u_hz: FloatArray = (x * cos_theta + y * sin_theta) * obs_x_mhz
    v_hz: FloatArray = (-x * sin_theta + y * cos_theta) * obs_y_mhz
    return u_hz, v_hz


__all__ = ["rotate_ppm_offsets"]
