# Headless Python fitting reference

The supported public import path is `peakfit.api`. Arrays and native files converge
on the same normalization, extraction, preparation, clustering, and fitting path.
`fit_spectrum` does not launch a GUI or require spectrum files, peak-list files,
output directories, temporary files, or continuation pickles. It uses the existing
PeakFit optimizer and `FinalFitOutcome`, with shared positions and widths and
independent signed amplitudes for each series member.

## Entry points

```python
from peakfit.api import (
	AxisProcessing, PeakFitConfig, PeakSeed, SeriesAxis, SpectrumInput,
	extract_spectrum, fit_file, fit_spectrum, model_availability,
	plan_extraction, prepare_spectrum,
)

source = SpectrumInput.from_array(data, udic, direct_axis=2)
result = fit_spectrum(
	source,
	peaks=[PeakSeed("stable-id", {0: 55.0, 1: 120.0, 2: 8.0})],
	region_ppm={0: (50.0, 65.0), 1: (110.0, 130.0), 2: (7.0, 9.0)},
	noise=0.1,
	config=PeakFitConfig(),
	output_dir=None,
)
```

These positions and bounds illustrate the interface, not a scientific fixture.
Runnable generated examples are in `examples/python-api/`:

```bash
uv run examples/python-api/fit_array.py
uv run examples/python-api/fit_series.py
uv run examples/python-api/fit_file_region.py --help
```

The first two examples need no input or output files. The third accepts a native
spectrum path and one `--region AXIS:LOW:HIGH` per frequency axis.

| Function | Arguments and return |
| --- | --- |
| `SpectrumInput.from_array` | `data, udic, *, direct_axis, series_axis=None, series=None, processing=None` → input descriptor |
| `SpectrumInput.from_file` | `path, *, direct_axis=None, series_axis=None, series=None, processing=None` → input descriptor |
| `model_availability` | `source, config=None` → per-axis selections, available models, missing fields, and reasons |
| `plan_extraction` | `source, region_ppm=None, *, halo_points=0, workers=1, peak_count=1, max_memory_bytes=None, exclude_planes=()` → slices, metadata, and resource estimate without a data copy |
| `extract_spectrum` | Same selection arguments → `ExtractedSpectrum`, containing `spectra` and `extraction` |
| `prepare_spectrum` | `source, *, peaks=None, peaklist_path=None, region_ppm=None, config=None, auto_pick=False, halo_points=0, noise=None, noise_region_ppm=None, workers=1, max_memory_bytes=None, cancel=None, progress=None` → `PreparedFit` without fitting |
| `fit_spectrum` | Preparation arguments plus `output_dir=None, optimizer="varpro"` → `SpectrumFitResult` |
| `fit_file` | `path, *, direct_axis=None, series_axis=None, series=None, processing=None, **fit_options` → the same completed result; `FitOptions` types the forwarded fitting arguments |

Public functions and returned public fields are compatibility contracts. Additive
fields may appear. Incompatible changes require release notes and migration
instructions. Private names and the internal `LoadedData`/`Spectra` preparation
containers are implementation details. Preparation containers are mutable domain
objects; completed scientific values come only from `result.outcome`.

Invalid inputs raise `ValueError` or `TypeError`. Native file failures may raise
`OSError`. Resource rejection raises `MemoryError`; cancellation raises
`FitCancelledError`. Model errors identify the external axis and missing history.
An unusable numerical fit remains a completed result with its actual classification
and failure diagnostics. Overall optimizer convergence does not substitute for
checking each cluster's usability.

## Array coordinates and ownership

`data` must be a nonempty real numeric NumPy ndarray. Object, complex, hypercomplex,
time-domain frequency axes, and nonfinite intensities are rejected. Float32 and
float64 are preserved through extraction; integer intensities become float64.
Read-only arrays and noncontiguous views are supported.

UDIC integer keys `0..ndim-1` refer to the supplied ndarray axes. Each axis requires
an exact integer `size`, a string `label`, and boolean `freq`, `time`, and `complex`.
Every frequency axis requires finite `sw` in Hz, positive `obs` in MHz, and `car`
in Hz. `sw` must be positive. Missing and generic labels such as `X`, `Y`, or
`unknown` are rejected. These checks cannot establish trustworthy referencing.
The caller remains responsible for labels, isotope identity, units, and calibration.

One, two, and three frequency axes are supported, with an optional explicit series
axis. Rank alone never establishes a series. `direct_axis` is required for arrays.
Frequency axes retain their external order except that the direct axis moves last.
The internal layout is always `(series, frequency..., direct)`. Data without a
series receive a singleton leading series axis. Isotope names never identify axes.
`extraction.axis_permutation` maps the external axes to the internal order;
`AxisExtraction.internal_axis` includes the leading series axis.

`from_array` borrows the array until preparation; do not modify it concurrently.
Metadata and peak records are copied immediately. Extraction owns a contiguous
copy of only the selected data. A completed result retains that crop and detached
final evaluations and model geometry. Later changes to caller arrays, dictionaries,
seeds, or configuration cannot alter completed results. There is no unsafe
zero-copy result option. The implementation does not downgrade float64 to float32.

## Series coordinates

```python
series = SeriesAxis(
	values=(20, 5, 20, 10), unit="ms", kind="relaxation",
	ids=("a", "b", "replicate-a", "c"),
)
source = SpectrumInput.from_array(
	data, udic, direct_axis=3, series_axis=1, series=series,
)
```

The designated series axis must not be marked as frequency data. It requires the
exact number of values, explicit units, and a supported kind. Time and relaxation
use `s`, `ms`, or `us`; CEST uses `Hz` or `ppm`; replicates use `index`. IDs are
unique stable member identities, generated from original indices if omitted.
Values may repeat. Their order and units remain unchanged; no sorting, kinetic
model, or intensity axis is introduced. `config.exclude_planes` refers to original
series indices. `extraction.series_indices` and the retained `SeriesAxis` preserve
the remaining member associations.

## Selection, padding, seeds, and noise

`region_ppm=None` selects every stored point. Otherwise the mapping must contain
exactly one pair for every external frequency axis and no series or unknown axes.
Duplicate CLI entries are rejected. A Python mapping cannot retain duplicate keys;
callers must reject duplicates before constructing their mapping if necessary.
Bounds may be reversed. Selection includes sampled coordinates in the closed
interval with an absolute `1e-9` ppm boundary tolerance. It uses no nearest-point
rounding or modulo wrapping. Empty, nonfinite, outside, and folded/aliased requests
are rejected. There is no implicit clipping mode. Native float32 header references
can differ slightly from an earlier UDIC; use the native coordinate grid when an
exact endpoint matters.

Slices are start-inclusive and stop-exclusive. Cropping preserves each original
sample coordinate and spacing, with consistently recomputed `size`, `sw`, and
`car`. `extraction.cropped_udic` describes the cropped external axes;
`native_headers` describes the original file. Acquisition time, apodization,
original TD/FFT lengths, and native headers do not become cropped acquisition
history. Indirect profiles retain the original stored folding period; segmentation
does not join opposite edges of a cropped axis. Automatic-picking zero-fill
fractions retain the original processed grid size.

`halo_points=0` is the default. Positive padding includes neighboring samples on
every frequency axis. Padding may clip at stored boundaries and reports that
fact. Each axis records requested bounds, actual analysis samples, actual fitting
samples, analysis/fitting slices, clipped padding, and neighbor expansion.

`PeakSeed` accepts `id`, `positions_ppm`, optional `widths_hz`, `sign`, `label`, and
`provenance`. IDs must be unique, nonempty, contain no periods, and not begin
`cluster_`, because existing parameter identifiers use dot separators and that
prefix. Positions must name every external frequency axis. Widths are positive
FWHM starting values in Hz, applied through the existing parameter constraints.
`sign` records an expected sign; it does not constrain the linear signed solve.

Seeds within the fitting extent are included. A seed outside it can expand the
fitting extent when its supplied width places it within four FWHM on every outside
axis. This finite neighbor rule is explicit; Lorentzian tails are not mathematically
zero beyond four FWHM. Without supplied widths, outside seeds are excluded with a
reason and an instruction to increase the halo. No seed outside the stored grid
wraps into the crop. Inspect `result.seeds` for inclusion, analysis membership,
reasons, boundary axes, labels, and provenance. All included neighbors contribute
to fitting; `PeakEstimate.in_analysis_region` distinguishes requested peaks.
No usable seeds is an error. `peaklist_path` accepts existing PeakFit peak-list
formats, in internal frequency-axis order. Existing unrestricted CLI behavior
remains compatible.

Automatic picking is experimental and requires `auto_pick=True` when seeds and
peak-list paths are absent. Supplied seeds never trigger it. Noise defaults to an
estimate from the fitting crop, reported explicitly. Peak-filled crops can bias
that estimate. Supply `noise`, `config.noise_level`, or an all-frequency-axis
`noise_region_ppm` from the source. Explicit noise must be positive and finite.
Conflicting noise sources are rejected.

## Processing capabilities

| Source | Empirical Gaussian/Lorentzian/pseudo-Voigt | Processing-aware models |
| --- | --- | --- |
| Processed Bruker | Supported with valid coordinates | Only with complete native acquisition/processing history |
| NMRPipe | Supported with valid coordinates | Only with complete retained history; EXT headers alone do not establish original acquisition timing |
| UCSF/Sparky | Supported with valid coordinates | Unavailable from UCSF headers alone |
| Array and UDIC | Supported with valid coordinates | Unavailable from UDIC alone |
| Array or file with separate `AxisProcessing` | Supported | Only models with complete validated requirements |

`AxisProcessing`, keyed by external frequency axis, accepts `provenance`,
`acquisition_time` in seconds, `td_size`, `ft_size`, `apod_code`, `apod_q1`,
`apod_q2`, `apod_q3`, `phase0_deg`, `phase1_deg`, `dqfcosy_window`, and
`processing_window`.
Processing metadata cannot be inferred from a filename, cropped grid, phase fields,
or a universal dictionary. NMRPipe and Bruker window codes are different; explicit
`AxisProcessing.apod_code` uses PeakFit/NMRPipe codes. Code `0` means unapodized;
code `1` with sine power `1` or `2` describes the supported sine window. Other
Bruker windows keep their separately validated DQF window if available and do not
claim SP1/SP2 support.

The phased no-apodization and sine models require acquisition time, original TD/FFT
sizes, a known window, and aggregate phase history. SP1/SP2 additionally require
complete sine coefficients with `0 <= q1 < q2 <= 1` and matching power. DQF-COSY
requires its existing validated `DQFCOSYWindow` per axis, either supplied explicitly
in configuration or processing metadata. Tilted fitting requires exactly two
frequency axes. DQF-COSY requires exactly two axes and `tilted=False`.
The real-array interface rejects `use_imaginary`; existing joint-quadrature CLI
workflows remain supported independently.

Set `config.fitting.include_window_functions=True` to fit intrinsic Gaussian,
Lorentzian, or pseudo-Voigt profiles through the recorded acquisition window.
This uses the same model as the CLI's `--include-window-functions`. It requires
one or two frequency axes and a validated `processing_window` for every axis.
`auto` selects intrinsic `pvoigt` in this mode. Missing window metadata raises
before cropping. Native Bruker and NMRPipe readers provide windows when their
headers contain the required fields. Arrays can supply them separately:

```python
from peakfit.api import AxisProcessing, ProcessingWindow

processing = {
    0: AxisProcessing(
        provenance="recorded acquisition and exponential processing",
        processing_window=ProcessingWindow(
            kind="em", points=128, dwell=0.001, duration=0.128, q1=5.0,
        ),
    ),
}
```

`points` is the original acquired complex-point count, `dwell` and `duration`
are in seconds, and coefficients follow the recorded window convention. See
[processing-aware fitting](processing-aware-fitting.md) for supported windows
and phase restrictions. Cropping preserves this object unchanged, including
when a saved region is reopened in the browser.

`auto` selects the supported processing model or empirical `pvoigt`, recording the
selected model, missing fields, and reason. Explicit unsupported requests raise
before cropping/fitting instead of silently switching. `model_availability(source)`
exposes this information without fitting or copying data. Separate metadata may
accompany UCSF-derived arrays or files with its own provenance; that does not make
UCSF headers a source of acquisition history.

## Completed values and exports

`SpectrumFitResult.outcome` is the existing immutable `FinalFitOutcome`. It contains
shared nonlinear parameters and standard errors, cluster classifications,
optimizer termination diagnostics, signed per-series amplitudes, the actual
amplitude covariance/error arrays, raw/normalized residuals, and fit statistics.
`result.peaks` projects only those authoritative values into original external-axis
ppm positions and stable seed IDs. Unusable clusters supply no invented amplitude
or position values. Neighbor decisions and failed cluster diagnostics remain visible.

Existing amplitude errors have shape `(peaks,)`, and covariance has shape
`(peaks, peaks)`. They are conditional on the fitted profiles and supplied noise;
scaled errors additionally use the existing reduced-chi-squared scale. The API
does not broadcast these into fictional `(peaks, series)` errors.
`PeakEstimate.amplitude_standard_errors` is `None` unless the authoritative
array actually has matching per-series shape. Use `outcome` for the shared errors.
Nonlinear standard errors retain existing covariance semantics; they do not become
posterior intervals. Amplitudes multiply the selected profile. Maximum heights and
integrated volumes are different quantities and are not relabeled as amplitudes.

`result.simulate()` evaluates usable final values only on the cropped internal
grid. `result.residual()` returns experimental minus simulated in intensity units.
Both return `None` if there are no usable clusters. They never allocate an
original-size spectrum. The extraction map supplies the source association.

`result.export(path)` or `output_dir=path` writes canonical `summary/fit.json` and
CSV tables using the existing writers, plus `metadata/extraction.json`. The sidecar
contains source headers, cropped metadata, coordinate mapping, series units and
IDs, seeds, model decisions, and diagnostics. Native byte fields are encoded as
hex records. No continuation pickle is required. With `config.output.save_simulated`,
`simulated_region.npz` contains cropped simulation and residual arrays. It is a
numeric export, not a new spectrum input format.

For two-frequency data, view an exported fit against its original supported
spectrum file:

```bash
peakfit plot spectrum --spectrum ./spectrum.ft2 --results ./regional-fit
```

The browser restores the recorded crop and series order, then evaluates the
final summary's parameters and amplitudes. It offers experimental, simulated,
and residual layers without refitting or a continuation pickle. The source file
must have the original shape and frequency grid. Array-only exports need a
matching supported source file to use this viewer. The browser supports two
frequency dimensions with optional series planes; true three-frequency browsing
is not supported. The viewer currently loads the full source before slicing.

## Resources, progress, and cancellation

`plan_extraction` reports selected points, owned crop bytes, estimated worker
workspace, optional simulation/residual bytes, and an estimated peak allocation.
Automatic-picking estimates allow up to one center per selected sample, bounded
by explicit ROI/peak limits when configured. Workers default to one; negative/all-CPU counts are rejected by the Python API.
Supply a positive `workers` count and optionally `max_memory_bytes` to reject the
estimated job before allocating a crop. This is a conservative workspace estimate,
not an operating-system memory guarantee. For a direct plan, `peak_count` must
reflect the expected number of fitted profiles. Native headers, interpreter state,
and caller-owned source arrays remain additional resident costs. JSON and CSV
size depends on result cardinality; `output_bytes` estimates numeric grid outputs.

NMRPipe and UCSF use nmrglue bounded access when available. NMRPipe 1D loading and
processed Bruker currently load their full data before slicing; `extraction.loading`
reports this. Bruker scaling occurs once in its native reader. The reader now preserves
nmrglue precision instead of forcing float32, which can increase resident memory
for existing Bruker CLI runs. No post-load slice
is presented as avoiding that full load.

`progress(event)` receives a resource estimate before crop allocation, then the
existing `ProgressStart` and cluster `FitResult` events. `cancel()` returns a
boolean and is checked before preparation, before fitting, at automatic-picking
ROI boundaries, and at progress boundaries. Raising `FitCancelledError` produces no result or exports. Cancellation
is cooperative between pipeline steps, not an interruption inside a SciPy solver.
Callbacks must not mutate internal fitting data. Worker pools close on unwinding.

## Regional CLI

```bash
peakfit fit spectrum.ucsf peaks.list --direct-axis 2 \
	--region-ppm 0:50:65 --region-ppm 1:110:130 --region-ppm 2:7:9 \
	--halo-points 2 --lineshape pvoigt --noise 0.1 --headless
```

All region axes are original file ndarray axes. The existing `--config`, lineshape,
constraint, optimizer, refinement, noise, peak-list, and output options apply.
For a series, add `--series 0:relaxation:ms --z-values times.txt` and provide bounds
only for frequency axes. UCSF requires explicit `--direct-axis` because its header
does not identify acquisition order. Region mode is headless and does not support
the optional GUI auto-pick stepper. It defaults to one worker when the legacy
worker default is unspecified/all-CPU; pass a positive `--workers` explicitly for
parallel work. Without `--region-ppm`, the existing CLI workflow is unchanged.
Regional output uses the same public exports and does not promise continuation
pickle files.

## External integration contract

An external caller sends an ndarray and truthful UDIC, or a processed path;
explicit axis/series identities; all-axis ppm bounds; stable peak seeds or opt-in
picking; noise input; and the existing `PeakFitConfig`. It receives an immutable
completed outcome, original-coordinate peak projections, ordered amplitudes,
classifications/uncertainty/residual diagnostics, and the extraction/model map.
No caller must construct PeakFit clusters, fake NMRPipe headers, or write temporary
input files. An independent CLI process still needs a supported processed file.

## Verification recorded on 2026-10-24

The existing uv environment and lockfile were used with `--no-sync`. Commands
used `UV_CACHE_DIR=/tmp/peakfit-uv-cache` in the restricted workspace.

- `uv run --no-sync pytest -q tests/test_headless_api.py`: 51 passed.
- `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run --no-sync pytest -q --ignore=tests/integration`: 424 passed.
- `uv run --no-sync ruff check .`: passed.
- `uv run --no-sync ruff format --check .`: 172 files formatted.
- `uv run --no-sync ty check --error-on-warning`: passed.
- `uv run --no-sync lint-imports`: both contracts kept.
- `UV_NO_SYNC=1 PREK_HOME=/tmp/peakfit-prek uv run --no-sync prek run --files ...`:
  all applicable hooks passed on the 22 changed files.
- `uv build --offline --no-build-isolation --out-dir /tmp/peakfit-headless-build`:
  wheel and source distribution built. The installed uv version warned that it
  lies outside the existing uv-build requirement range; neither build failed.
- Both generated array examples ran and recovered their declared amplitudes.
  The file example's `--help` ran without input data.

The two experimental-example integration modules were excluded to avoid loading
unrelated experimental datasets. `prek run --all-files` could not finish because
its end-of-file fixer tried to write a pre-existing read-only skill file at
`.agents/skills/setup-matt-pocock-skills/triage-labels.md`. That file was preserved.
Project-wide Ruff, typing, and architecture checks still ran independently.
The remaining runtime limits are documented above: complete metadata is required
for processing models, cancellation occurs between solver steps, memory estimates
are budgets rather than OS guarantees, and Bruker currently loads fully.
