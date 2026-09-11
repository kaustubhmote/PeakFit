# Carry tilt through post-fit workflows

Type: task
Status: ready-for-agent
Blocked by: 03, 04

## Objective

Keep the fitted geometry intact after optimization.

## Work

- Verify the final outcome, JSON, and CSV contain the tilt parameter.
- Preserve tilt in the immutable simulation snapshot and saved fitting state.
- Make MCMC include tilt through the normal nonlinear-parameter path.
- Update reconstruction so a cropped grid cannot clip a rotated peak. Use the
  full grid first if that is the smallest correct change.
- Exercise tilted automatic peak picking without changing its default mode.

## Done when

Saved results reconstruct the same tilted surface that the optimizer fitted,
and post-fit workflows do not need a second model implementation.

## Comments
