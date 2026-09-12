# Document and verify tilted fitting

Type: task
Status: resolved
Blocked by: 05

## Objective

Document the user workflow, limits, and interpretation, then run the full
project checks.

## Work

- Add concise CLI and TOML examples to the fitting documentation.
- List the supported families and explain that `auto` remains compatible.
- Define the ppm-space angle convention and the isotropic-peak limitation.
- State the processed-profile and doublet caveats from the spec.
- Add end-to-end synthetic recovery tests for Gaussian, pseudo-Voigt, and one
  processed profile family.
- Run the headless suite, Ruff lint and format checks, the type checker, and
  import contracts.

## Done when

The documented command works, all acceptance criteria in the spec have tests,
and the repository checks pass.

## Comments

The README now documents the CLI, TOML, parameter constraints, supported
profiles, angle convention, and interpretation limits. Synthetic VarPro tests
recover known tilt for Gaussian, pseudo-Voigt, and no-apodization profiles.
