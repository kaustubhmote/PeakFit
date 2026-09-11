# Lock the tilted-model contract

Type: task
Status: resolved

## Objective

Add pure numerical tests for the ppm-space rotation in the spec before
changing the fitting pipeline.

## Work

- Add a small pure rotation function under `engine.lineshapes`.
- Test zero angle, positive and negative angles, and observation-frequency
  conversion.
- Add a closed-form rotated-Gaussian fixture.
- Record the doublet and processed-profile semantics in test names.

## Done when

The mathematical contract is executable without constructing a CLI fit.

## Comments

Implemented in the first numerical slice. The tests use literal zero-degree,
45-degree, mirrored-angle, and closed-form Gaussian reference values.
