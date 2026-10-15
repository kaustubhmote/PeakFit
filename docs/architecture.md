# PeakFit Architecture Guide

This is a short map of the current repository, not an immutable contract. `AGENTS.md`
is the source of truth for AI-assisted work and explicitly allows architecture changes
when they simplify the project while preserving scientific correctness and useful
workflows.

## Purpose

PeakFit fits lineshape models to pseudo-ND NMR spectra through a Python package and CLI.
The important domains are spectra and peak inputs, clustering, lineshapes, optimization,
fit orchestration, uncertainty analysis, plotting, and output generation.

## Current Package Map

- `src/peakfit/cli/` - Typer command-line entrypoints and terminal presentation.
- `src/peakfit/ui/` - Rich terminal helpers used by the current CLI.
- `src/peakfit/fit/` - validation, data loading, fitting workflow, result assembly, and output coordination.
- `src/peakfit/auto_pick/` - experimental automatic peak-picking workflow used only when no peak list is provided.
- `src/peakfit/mcmc/` - uncertainty sampling, diagnostics, and MCMC-facing workflows.
- `src/peakfit/plot/` - plotting and the spectrum viewer.
- `src/peakfit/io/` - parsing, serialization, and file-format handling.
- `src/peakfit/engine/` - domain models, numerical algorithms, lineshapes, and fitting math.
- `src/peakfit/shared/` - small cross-cutting helpers.

## Current Flow

1. `cli` parses user intent and builds configuration. For Bruker input without an
   explicit peak list, it discovers `peaklist.xml` in the resolved processing folder
   before validation. The XML parser in `io` supplies both validation and fitting.
   Generic Bruker acquisition duration uses the acquisition sweep width, even
   when the frequency coordinates describe an extracted spectrum. Other Bruker
   windows currently fall back to empirical pseudo-Voigt profiles; this does
   not supply an exact Gaussian-multiplication or linear-prediction model.
2. `fit` validates inputs, loads spectra/peaks, prepares clusters, and runs direct fit pipeline functions.
   When no peak list is provided or discovered, `fit` delegates to the experimental `auto_pick` package instead of owning that logic.
3. `engine` performs numerical work: clustering, lineshapes, optimizers, residuals, and statistics helpers.
   One-dimensional lineshape profiles evaluate supplied frequency offsets.
   `Cluster.evaluate_peaks` owns multidimensional peak geometry, including the
   optional F2-F3 rotation and its analytic derivatives.
   DQF-COSY profiles share one `F2-F3.j` parameter and have separate intrinsic
   widths and fixed processing windows. The Bruker reader derives these windows
   from acquisition and processing dictionaries; per-axis configuration overrides
   the detected settings. Their fitting rectangles include the
   marked zero-signal midpoint and all four lobes. Rotation and automatic peak
   picking are unavailable for this model. `--use-imaginary` stacks two to four
   quadrature planes. Clustering expands their spatial coordinates into
   component-major observations with an explicit component index; each cluster
   has one amplitude series shared across all components. DQF-COSY evaluates the
   correct real or imaginary profile on each axis, preserving analytic shared-J
   derivatives. The existing real least-squares and MCMC paths then operate on
   all observations together. Simulation snapshots retain component identities
   and reconstruct all planes. JSON metadata identifies the component order;
   the intensity table records one shared amplitude. See [`dqfcosy.md`](dqfcosy.md).
4. `fit.final_outcome` freezes completed scientific truth by run-local
   `cluster_id` against the terminal correction revision, while
   `fit.output_metadata` captures non-scientific run metadata; `io` reads inputs
   and projects the outcome into structured artifacts.
5. `mcmc` and `plot` run post-fit workflows from canonical fit outputs.
6. `ui` renders terminal output where the CLI needs it.

## Browser spectrum viewer

`plot.spectra_viewer` loads two spectral dimensions plus a plane axis through the
shared spectrum reader. A local HTTP server serves bundled HTML and JavaScript;
there is no JavaScript build step, external service, or CDN. ContourPy extracts
positive and negative full-resolution contour paths. The server sends little-endian
float32 coordinates with NaN path breaks. Canvas caches paths and transforms them
for zoom and pan without further numerical work or network requests. A shared
plot rectangle defaults to a centered square, with selectable aspect ratios and
a fit-window option. Drawing, axes, pointer coordinates, and panning all use that
rectangle; changing the ratio preserves the viewed sample ranges. Both sides
bound their contour caches to a 64 MiB budget; the browser estimates native path
memory. The server retains only one simulated plane at a time.

`plot.reconstruction` uses the final nonlinear parameters and retained amplitudes
from `summary/fit.json`, matched by cluster ID and peak order to geometry in
`metadata/fitting_state.pkl`. It evaluates complete profiles in point blocks,
including tails and tilted peaks. It excludes unusable outcomes and checks the
input ppm grid and plane count before serving. The residual is experimental minus
simulated, and all layers share signed contour levels. The current viewer supports
the real `rr` plane of joint quadrature fits.

The CLI imports Matplotlib output generators only for PDF commands. The browser
viewer imports neither Matplotlib nor Qt. The experimental automatic picker still
uses its existing Qt and Matplotlib widgets, and PDF output still uses Matplotlib.
These dependencies therefore remain in the package.

## Boundaries

- `engine` should stay free of Typer, Rich, Qt, and filesystem side effects.
- `fit`, `mcmc`, and `plot` may orchestrate workflows, but should call direct
  functions rather than adding manager/service layers.
- `io.schemas` is the JSON validation/read contract. `FinalFitOutcome` is the
  sole completed scientific result; `FittingState` is continuation-only and
  must not be used to reconstruct completed output.
- `auto_pick` is intentionally separate from `fit` because it is experimental.

## Invariants To Preserve Or Change Explicitly

- Numerical and scientific behavior should remain correct and tested.
- Expensive fit work should fail early on invalid inputs.
- Input parsing errors should be clear and actionable.
- Long-running jobs should show understandable progress and failures.
- Pure numerical code should not accidentally depend on terminal UI, Typer, Qt, or filesystem side effects.
- Peak-cluster data has the strict shape `(n_points, n_series)`; point, series,
  observation, and amplitude counts derive from that one representation.
- Public behavior changes should be documented with migration notes when needed.

## Simplification Notes

The current vertical-slice shape is a starting point. It is acceptable to merge packages,
rename modules, delete thin wrappers, or move responsibilities when that makes the code
easier to understand and test. Prefer direct data flow and domain names over generic
services, managers, factories, adapters, or registries.
