# Tilted two-dimensional peak shapes

## Goal

Let a user fit a rotated two-dimensional peak with any lineshape that PeakFit
already supports. Keep the one-dimensional lineshape choice separate from the
two-dimensional geometry:

```bash
peakfit fit spectrum.ft2 peaks.list --lineshape pvoigt --tilted
```

The equivalent TOML setting is:

```toml
[fitting]
lineshape = "pvoigt"
tilted = true
```

The existing axis-aligned model remains the default.

## Why the current design cannot express a tilt

`Cluster.evaluate_peaks` evaluates one `Shape` per spectral dimension and
multiplies the results. That construction can produce only axis-aligned peak
surfaces. Adding `tilted_gaussian` and similar names to `SHAPES` would still
evaluate each axis independently, so it would not implement a rotated peak.

Tilt belongs to the joint two-dimensional peak geometry. Gaussian,
Lorentzian, pseudo-Voigt, SP1, SP2, and no-apodization remain the
one-dimensional profile families.

## Supported combinations

The `--tilted` option applies to all twelve registered lineshapes:

| Profile family | Singlet | Doublet |
| --- | --- | --- |
| Gaussian | `gaussian` | `gaussian_doublet` |
| Lorentzian | `lorentzian` | `lorentzian_doublet` |
| Pseudo-Voigt | `pvoigt` | `pvoigt_doublet` |
| Sine bell | `sp1` | `sp1_doublet` |
| Squared sine bell | `sp2` | `sp2_doublet` |
| No apodization | `no_apod` | `no_apod_doublet` |

`auto` remains a selector. `--lineshape auto --tilted` detects the profile for
each dimension and then applies the tilted geometry.

## Mathematical contract

Define offsets from the fitted center in ppm:

\[
x = \delta_{F2} - \delta_{F2,0}, \qquad
y = \delta_{F3} - \delta_{F3,0}.
\]

For a fitted angle \(\theta\), measured counterclockwise in the F2-F3 ppm
plane, define:

\[
u = x\cos\theta + y\sin\theta,
\]

\[
v = -x\sin\theta + y\cos\theta.
\]

Convert the rotated offsets to Hz with the observation frequency associated
with each profile:

\[
\Delta\nu_u = \mathrm{obs}_{F2}u, \qquad
\Delta\nu_v = \mathrm{obs}_{F3}v.
\]

The tilted peak is:

\[
P_\theta(x,y) = K_{F2}(\Delta\nu_u)K_{F3}(\Delta\nu_v),
\]

where each \(K\) is the selected existing one-dimensional kernel with its
current linewidth, eta, phase, apodization, and doublet parameters.

This definition has three useful properties:

- `tilt = 0` is the current model.
- Peak height normalization at the center does not change.
- The angle has the same meaning across magnetic fields because the rotation
  uses ppm before each profile converts back to Hz.

For `gaussian`, the result is the rotated Gaussian discussed in the referenced
study. For the other families, the same coordinate rotation defines a tilted
product profile. SP1, SP2, and no-apodization are phenomenological when
rotated. Their window and phase parameters remain attached to their original
profile families.

The `_doublet` variants rotate the complete current two-dimensional doublet
surface. This definition gives an exact rotated form of the existing model,
but it does not claim that scalar-coupling displacements rotate in a physical
NMR experiment. A model that keeps J splitting on the laboratory axes while
tilting only the envelope would be a separate model.

## Parameter contract

Each peak gets one joint parameter:

```text
<peak-name>.F2-F3.tilt
```

The parameter has units of degrees, starts at `0.0`, and has default bounds of
`-89.0` to `89.0`. The axis pair uses the spectral dimension order in the
input. Configuration constraints can override the start, bounds, or vary
flag:

```toml
[parameters.defaults]
"*.*.tilt" = { min = -60.0, max = 60.0 }

[parameters.peaks."G45N-HN"]
"F2-F3.tilt" = { value = 35.0, vary = true }
```

Angles are not identifiable for an isotropic peak. Width and angle can also
trade off for weak or heavily overlapped peaks. PeakFit must report the fitted
angle and uncertainty without presenting the angle as reliable in those
cases. Users can compare the axis-aligned and tilted runs with the existing
AIC and BIC output.

## Scope

The first implementation supports exactly two spectral dimensions. A
pseudo-3D series has two spectral dimensions plus one intensity-series axis,
so it is supported. One-dimensional spectra and data with more than two
spectral dimensions fail before peak picking or fitting with an actionable
message.

The first implementation does not add:

- automatic selection between axis-aligned and tilted models;
- a shared angle across peaks or clusters;
- a two-population diagonal pair;
- a laboratory-axis J-doublet with a separately tilted envelope;
- tilt between the pseudo dimension and a spectral dimension.

The two-population pseudo-Voigt model from the referenced study should use a
future joint component model. It must not reuse `_doublet`, which currently
means J splitting.

## Target design

Keep `create_shapes` and the existing lineshape registry responsible for
selecting one-dimensional profiles. Do not add twelve tilted wrapper classes.

Move multidimensional coordinate handling into the cluster evaluation path:

1. `Peak` owns whether its geometry is axis-aligned or tilted and creates its
   joint `F2-F3.tilt` parameter.
2. Each `Shape` evaluates a supplied Hz-offset matrix and returns derivatives
   with respect to that offset and its own parameters.
3. `Cluster.evaluate_peaks` computes axis offsets, applies the rotation, calls
   the selected profiles, multiplies their values, and applies the chain rule.
4. The same evaluator remains the single source for optimization, analytical
   evaluation, simulation, plotting, and MCMC.

This design puts the joint geometry at the only seam that has both coordinate
axes. The profile interface stays small, and every existing kernel gains tilt
support through one implementation.

## Derivative requirements

The analytic Jacobian must cover:

- both chemical shifts, including their cross-axis effects after rotation;
- both linewidths;
- `eta`, `phase`, and `j` where the selected profile uses them;
- `tilt` through the rotated coordinates.

With angles converted to radians for evaluation:

\[
\frac{\partial u}{\partial\theta}=v, \qquad
\frac{\partial v}{\partial\theta}=-u.
\]

The implementation must include the degrees-to-radians factor in the tilt
derivative. Finite-difference tests protect every analytic derivative path.

## Downstream behavior

- `summary/fit.json` and `tables/parameters.csv` include the tilt parameter
  through the existing final-parameter projection.
- Run metadata records `fitting.tilted` through the existing configuration
  snapshot.
- Saved fitting state retains the joint geometry for MCMC and reconstruction.
- The full-grid and cropped-grid simulation paths produce the same values.
- Reconstruction either computes a rotated bounding box or uses the full grid
  until a correct cropped bound exists. It must not clip a tilted peak.
- Automatic peak picking may fit tilt only when the user enables it. The
  default workflow and F-test behavior do not change.

## Compatibility

This change is additive. Runs without `--tilted` keep their current parameter
names, numerical values, and output shape. No existing lineshape name changes.

The only new rejection is a request for `--tilted` on data that does not have
exactly two spectral dimensions.

## Acceptance criteria

1. `tilt = 0` matches the current evaluation for all twelve registered
   lineshapes within floating-point tolerance.
2. A tilted Gaussian matches the closed-form rotated Gaussian on a known grid.
3. Positive and negative angles produce mirrored surfaces.
4. Analytic derivatives match central finite differences for tilt, both
   chemical shifts, and every family-specific parameter.
5. Synthetic fits recover known angles for Gaussian, pseudo-Voigt, and one
   processed profile family.
6. CLI and TOML workflows reject unsupported dimensions before fitting.
7. Tilt parameters survive constraints, final-output projection, saved state,
   MCMC setup, and spectrum reconstruction.
8. The complete headless test suite, Ruff, formatting, type checks, and import
   contracts pass.

## Reviewable implementation order

Implement the work through the issues in `.scratch/tilted-peaks/issues/`.
Each issue is one intended commit. Do not combine the evaluator refactor, the
user interface, and downstream reconstruction into one commit.
