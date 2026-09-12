"""Domain model representing a cluster of peaks."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from peakfit.engine.domain.coordinates import rotate_ppm_offsets
from peakfit.engine.domain.peaks import Peak
from peakfit.shared.typing import FloatArray, IntArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peakfit.engine.domain.param_map import ParameterMap
    from peakfit.engine.domain.params_scalar import Parameters

_CLUSTER_DATA_NDIM = 2
_TILTED_DIMENSIONS = 2


@dataclass(slots=True)
class Cluster:
    """Grouped peaks sharing a contiguous spectral segment."""

    cluster_id: int
    peaks: list[Peak]
    grid_indices: list[IntArray]
    data: FloatArray
    corrections: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        """Validate point-by-series data and initialize calculated fields."""
        self.data = np.asarray(self.data, dtype=np.float64)
        self.grid_indices = [np.asarray(indices, dtype=np.intp) for indices in self.grid_indices]

        if self.data.ndim != _CLUSTER_DATA_NDIM:
            msg = (
                "Cluster data must be a two-dimensional "
                f"(n_points, n_series) array, got shape {self.data.shape}"
            )
            raise ValueError(msg)
        if not self.grid_indices:
            msg = "Cluster grid_indices cannot be empty"
            raise ValueError(msg)

        grid_point_counts = {indices.size for indices in self.grid_indices}
        if len(grid_point_counts) != 1:
            msg = "Cluster grid dimensions must contain the same number of points"
            raise ValueError(msg)

        grid_point_count = next(iter(grid_point_counts))
        if grid_point_count == 0:
            msg = "Cluster data must contain at least one point"
            raise ValueError(msg)
        if self.data.shape[0] != grid_point_count:
            msg = (
                f"Cluster data point count ({self.data.shape[0]}) does not match "
                f"spectral grid point count ({grid_point_count})"
            )
            raise ValueError(msg)
        if self.data.shape[1] == 0:
            msg = "Cluster data must contain at least one series"
            raise ValueError(msg)

        self.corrections = np.zeros_like(self.data, dtype=np.float64)

    @classmethod
    def from_clusters(cls, clusters: Sequence[Cluster]) -> Cluster:
        """Create a single cluster by merging a list of clusters."""
        if not clusters:
            msg = "clusters list cannot be empty"
            raise ValueError(msg)
        result = clusters[0]
        for other in clusters[1:]:
            result = result + other
        return result

    @property
    def corrected_data(self) -> FloatArray:
        """Return data with corrections subtracted (ready for processing)."""
        return self.data - self.corrections

    @property
    def n_points(self) -> int:
        """Return the number of sampled spectral grid points."""
        return self.data.shape[0]

    @property
    def n_series(self) -> int:
        """Return the number of spectra in the pseudo dimension."""
        return self.data.shape[1]

    @property
    def n_observations(self) -> int:
        """Return the number of scalar observations in the cluster."""
        return self.data.size

    @property
    def n_amplitude_params(self) -> int:
        """Return the number of amplitude parameters (DOF)."""
        return len(self.peaks) * self.n_series

    def evaluate(
        self,
        params: Parameters,
        grid_indices: Sequence[IntArray] | None = None,
        d_matrix: np.ndarray | None = None,
        param_map: ParameterMap | None = None,
    ) -> np.ndarray:
        """Evaluate shapes for all peaks on the provided grid positions."""
        grid = list(self.grid_indices) if grid_indices is None else list(grid_indices)
        return self.evaluate_peaks(
            self.peaks,
            params,
            grid,
            d_matrix=d_matrix,
            param_map=param_map,
        )

    @staticmethod
    def evaluate_peaks(
        peaks: Sequence[Peak],
        params: Parameters,
        grid_indices: Sequence[IntArray],
        *,
        d_matrix: np.ndarray | None = None,
        param_map: ParameterMap | None = None,
    ) -> np.ndarray:
        """Vectorized evaluation for peaks on provided grid positions.

        Returns a consistent shapes matrix of shape (n_peaks, n_points) for any
        n_peaks >= 0.
        """
        if not grid_indices:
            msg = "grid_indices cannot be empty"
            raise ValueError(msg)

        grid_dims: list[np.ndarray] | np.ndarray = list(grid_indices)
        if isinstance(grid_dims, np.ndarray):
            grid_dims = [grid_dims]

        n_peaks = len(peaks)
        n_points = grid_dims[0].size
        n_dims = len(grid_dims)
        compute_derivatives = d_matrix is not None and param_map is not None

        has_tilted_peak = any(peak.tilted for peak in peaks)
        has_nonzero_tilt = any(
            peak.tilted and params[peak.tilt_parameter_id.name].value != 0.0 for peak in peaks
        )
        if has_tilted_peak and (compute_derivatives or has_nonzero_tilt):
            return _evaluate_tilted_peaks(
                peaks,
                params,
                grid_dims,
                d_matrix=d_matrix,
                compute_derivatives=compute_derivatives,
                param_map=param_map,
            )

        # accum_val: (n_points, n_peaks) starting as ones
        accum_val = np.ones((n_points, n_peaks), dtype=np.float64)

        if n_peaks == 0:
            return accum_val.T

        dim_vals: list[np.ndarray] = []
        dim_deriv_maps: list[dict[str, FloatArray]] = []
        param_indices_map: list[dict[str, IntArray]] = []

        for d in range(n_dims):
            val, current_derivs, idx_map = _evaluate_single_dimension(
                d, peaks, params, grid_dims[d], param_map, compute_derivatives
            )

            dim_vals.append(val)
            accum_val *= val

            if compute_derivatives:
                dim_deriv_maps.append(current_derivs)
                param_indices_map.append(idx_map)

        if compute_derivatives and d_matrix is not None:
            for d in range(n_dims):
                others_prod = np.ones((n_points, n_peaks))
                for other_d in range(n_dims):
                    if other_d != d:
                        others_prod *= dim_vals[other_d]

                p_map = param_indices_map[d]
                deriv_map = dim_deriv_maps[d]

                for param_type, indices in p_map.items():
                    if param_type in deriv_map:
                        deriv_total = deriv_map[param_type] * others_prod
                        _accumulate_derivatives(d_matrix, indices, deriv_total)

        return accum_val.T

    def __add__(self, other: object) -> Cluster:
        """Concatenate two clusters, preserving peaks and data arrays."""
        if not isinstance(other, Cluster):
            return NotImplemented
        if self.n_series != other.n_series:
            msg = "Merged clusters must contain the same number of series"
            raise ValueError(msg)

        # Concatenate positions
        new_positions = [
            np.concatenate((positions_self, positions_other))
            for positions_self, positions_other in zip(
                self.grid_indices,
                other.grid_indices,
                strict=False,
            )
        ]

        # Concatenate data
        new_data = np.concatenate((self.data, other.data), axis=0)

        return type(self)(
            cluster_id=self.cluster_id,
            peaks=self.peaks + other.peaks,
            grid_indices=new_positions,
            data=new_data,
        )


def _evaluate_single_dimension(
    d: int,
    peaks: Sequence[Peak],
    params: Parameters,
    grid_dim: np.ndarray,
    param_map: ParameterMap | None,
    compute_derivatives: bool,
) -> tuple[FloatArray, dict[str, FloatArray], dict[str, IntArray]]:
    """Evaluate shapes for a single dimension.

    Uses the unified evaluation pipeline where the Shape class IS the evaluator.
    All parameters flow through ClusterParameters uniformly.
    """
    sample_shape = peaks[0].shapes[d]

    cluster_params = sample_shape.get_cluster_parameters(
        peaks, params, param_map=param_map if compute_derivatives else None
    )

    x_grid = grid_dim.ravel().astype(np.float64)
    result = sample_shape.evaluate_cluster(
        x_grid, cluster_params, compute_derivs=compute_derivatives
    )

    return result.values, result.derivatives, cluster_params.index_map


def _evaluate_tilted_peaks(
    peaks: Sequence[Peak],
    params: Parameters,
    grid_dims: list[np.ndarray],
    *,
    d_matrix: np.ndarray | None,
    compute_derivatives: bool,
    param_map: ParameterMap | None,
) -> np.ndarray:
    """Evaluate two-dimensional peaks after rotating their ppm offsets."""
    if len(grid_dims) != _TILTED_DIMENSIONS:
        msg = "Tilted peak evaluation requires exactly two spectral dimensions"
        raise ValueError(msg)

    shape_x = peaks[0].shapes[0]
    shape_y = peaks[0].shapes[1]
    cluster_params_x = shape_x.get_cluster_parameters(
        peaks,
        params,
        param_map=param_map if compute_derivatives else None,
    )
    cluster_params_y = shape_y.get_cluster_parameters(
        peaks,
        params,
        param_map=param_map if compute_derivatives else None,
    )
    offsets_x_hz, sign_x = shape_x.dim_ctx.compute_offsets(
        grid_dims[0].ravel().astype(np.float64),
        cluster_params_x.extras["cs"],
    )
    offsets_y_hz, sign_y = shape_y.dim_ctx.compute_offsets(
        grid_dims[1].ravel().astype(np.float64),
        cluster_params_y.extras["cs"],
    )
    obs_x_mhz = float(shape_x.dim_ctx.ppm2hz(1.0))
    obs_y_mhz = float(shape_y.dim_ctx.ppm2hz(1.0))
    tilt_degrees = np.array(
        [params[peak.tilt_parameter_id.name].value if peak.tilted else 0.0 for peak in peaks],
        dtype=np.float64,
    )
    rotated_x_hz, rotated_y_hz = rotate_ppm_offsets(
        offsets_x_hz / obs_x_mhz,
        offsets_y_hz / obs_y_mhz,
        tilt_degrees=tilt_degrees,
        obs_x_mhz=obs_x_mhz,
        obs_y_mhz=obs_y_mhz,
    )
    result_x = shape_x.evaluate_offsets(
        rotated_x_hz,
        sign_x,
        cluster_params_x,
        compute_derivs=compute_derivatives,
    )
    result_y = shape_y.evaluate_offsets(
        rotated_y_hz,
        sign_y,
        cluster_params_y,
        compute_derivs=compute_derivatives,
    )
    values = result_x.values * result_y.values
    if not compute_derivatives or d_matrix is None or param_map is None:
        return values.T

    _accumulate_tilted_profile_derivatives(
        d_matrix,
        result_x.derivatives,
        result_y.values,
        cluster_params_x.index_map,
    )
    _accumulate_tilted_profile_derivatives(
        d_matrix,
        result_y.derivatives,
        result_x.values,
        cluster_params_y.index_map,
    )

    derivative_x = result_x.derivatives["offset"] * result_y.values
    derivative_y = result_y.derivatives["offset"] * result_x.values
    theta = np.deg2rad(tilt_degrees)[None, :]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cs_x = derivative_x * (-obs_x_mhz * cos_theta) + derivative_y * (obs_y_mhz * sin_theta)
    cs_y = derivative_x * (-obs_x_mhz * sin_theta) + derivative_y * (-obs_y_mhz * cos_theta)
    _accumulate_derivatives(d_matrix, cluster_params_x.index_map["cs"], cs_x)
    _accumulate_derivatives(d_matrix, cluster_params_y.index_map["cs"], cs_y)

    u_ppm = rotated_x_hz / obs_x_mhz
    v_ppm = rotated_y_hz / obs_y_mhz
    tilt_derivative = (derivative_x * (obs_x_mhz * v_ppm) - derivative_y * (obs_y_mhz * u_ppm)) * (
        np.pi / 180.0
    )
    tilt_indices = np.full(len(peaks), -1, dtype=np.intp)
    for index, peak in enumerate(peaks):
        if not peak.tilted:
            continue
        mapped_index = param_map.get(peak.tilt_parameter_id.name, -1)
        tilt_indices[index] = -1 if mapped_index is None else mapped_index
    _accumulate_derivatives(d_matrix, tilt_indices, tilt_derivative)
    return values.T


def _accumulate_tilted_profile_derivatives(
    d_matrix: np.ndarray,
    derivatives: dict[str, FloatArray],
    other_dimension_values: FloatArray,
    index_map: dict[str, IntArray],
) -> None:
    """Accumulate one profile's own parameters into the joint Jacobian."""
    for parameter_name, indices in index_map.items():
        if parameter_name == "cs" or parameter_name not in derivatives:
            continue
        _accumulate_derivatives(
            d_matrix,
            indices,
            derivatives[parameter_name] * other_dimension_values,
        )


def _accumulate_derivatives(
    d_matrix: np.ndarray,
    indices: IntArray,
    deriv_total: FloatArray,
) -> None:
    """Accumulate derivatives for mapped parameters, supporting shared indices."""
    valid_mask = indices >= 0
    if not np.any(valid_mask):
        return

    np.add.at(d_matrix, indices[valid_mask], deriv_total[:, valid_mask].T)
