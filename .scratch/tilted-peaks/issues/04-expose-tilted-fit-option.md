# Expose the tilted fit option

Type: task
Status: ready-for-agent
Blocked by: 03

## Objective

Add the opt-in CLI and TOML interface without expanding the lineshape-name
list.

## Work

- Add `FitConfig.tilted`, defaulting to `false`.
- Add `--tilted` to `peakfit fit` and pass it through `build_fit_config`.
- Allow axis-pair keys such as `F2-F3.tilt` in per-peak constraints.
- Validate that tilted fits have exactly two spectral dimensions before peak
  picking or optimization.
- Add CLI-help, config round-trip, constraint, and error-message tests.

## Done when

Both command-line and configuration-file users can request a tilted fit, and
unsupported inputs fail early with a clear message.

## Comments
