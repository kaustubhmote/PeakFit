"""Shared helpers for doublet lineshapes built from singlet kernels."""

from typing import TYPE_CHECKING, Any

import numpy as np

from peakfit.engine.lineshapes.utils import doublet_offsets

if TYPE_CHECKING:
    from collections.abc import Callable


def doublet_kernel(
    x,
    cs,
    j,
    grid,
    *,
    kernel: Callable[..., Any],
    kernel_args: tuple[Any, ...] = (),
):
    """Combine +/- J/2 components for a singlet kernel."""
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    return sign_p * kernel(dw_p, *kernel_args) + sign_m * kernel(dw_m, *kernel_args)


def doublet_kernel_with_derivs(
    x,
    cs,
    j,
    grid,
    *,
    kernel_with_derivs: Callable[..., tuple[Any, dict[str, Any]]],
    kernel_args: tuple[Any, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    """Combine +/- J/2 components with derivative aggregation."""
    dw_p, sign_p, dw_m, sign_m = doublet_offsets(x, cs, j, grid)
    v_p, d_p = kernel_with_derivs(dw_p, *kernel_args)
    v_m, d_m = kernel_with_derivs(dw_m, *kernel_args)
    values = sign_p * v_p + sign_m * v_m
    derivs: dict[str, Any] = {}
    for key in d_p.keys() & d_m.keys():
        derivs[key] = sign_p * d_p[key] + sign_m * d_m[key]
    if "dw" in d_p and "dw" in d_m:
        derivs["j"] = -0.5 * sign_p * d_p["dw"] + 0.5 * sign_m * d_m["dw"]
    return values, derivs


def doublet_kernel_from_offsets(
    offsets_hz,
    sign,
    j_hz,
    *,
    kernel: Callable[..., Any],
    kernel_args: tuple[Any, ...] = (),
):
    """Evaluate a doublet from precomputed center-relative Hz offsets."""
    j_shift = 0.5 * np.asarray(j_hz, dtype=np.float64)[None, :]
    return sign * (
        kernel(offsets_hz - j_shift, *kernel_args) + kernel(offsets_hz + j_shift, *kernel_args)
    )


def doublet_kernel_from_offsets_with_derivs(
    offsets_hz,
    sign,
    j_hz,
    *,
    kernel_with_derivs: Callable[..., tuple[Any, dict[str, Any]]],
    kernel_args: tuple[Any, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    """Evaluate a doublet and derivatives from precomputed Hz offsets."""
    j_shift = 0.5 * np.asarray(j_hz, dtype=np.float64)[None, :]
    value_plus, derivs_plus = kernel_with_derivs(
        offsets_hz - j_shift,
        *kernel_args,
    )
    value_minus, derivs_minus = kernel_with_derivs(
        offsets_hz + j_shift,
        *kernel_args,
    )
    values = sign * (value_plus + value_minus)
    derivs = {
        key: sign * (derivs_plus[key] + derivs_minus[key])
        for key in derivs_plus.keys() & derivs_minus.keys()
    }
    if "dw" in derivs_plus and "dw" in derivs_minus:
        derivs["j"] = 0.5 * sign * (derivs_minus["dw"] - derivs_plus["dw"])
    return values, derivs


__all__ = [
    "doublet_kernel",
    "doublet_kernel_from_offsets",
    "doublet_kernel_from_offsets_with_derivs",
    "doublet_kernel_with_derivs",
]
