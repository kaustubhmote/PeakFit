# Add joint tilt evaluation

Type: task
Status: ready-for-agent
Blocked by: 02

## Objective

Make `Cluster.evaluate_peaks` evaluate either the current axis-aligned geometry
or the tilted two-dimensional geometry through one code path.

## Work

- Let `Peak` create and retain `<peak>.F2-F3.tilt` when tilt is enabled.
- Rotate ppm offsets before converting each principal coordinate to Hz.
- Vectorize over both grid points and peaks.
- Accumulate analytic derivatives for tilt and both cross-coupled chemical
  shifts into the existing VarPro derivative matrix.
- Preserve family-specific linewidth, eta, phase, and J derivatives.
- Test all twelve profiles at zero and nonzero tilt.

## Done when

Optimization and analytical evaluation use one joint evaluator, and its
Jacobian matches finite differences.

## Comments
