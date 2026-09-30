# Fit HN-Hα couplings from DQF-COSY

Use `dqfcosy` for isolated, phase-sensitive 2D proton cross-peaks with four
antiphase lobes. Put one peak-list entry at the approximate midpoint of the four
lobes. PeakFit fits that midpoint, two intrinsic linewidths, and one positive
coupling magnitude shared by both axes. J fitting is automatic for this model.

```bash
uv run peakfit fit spectrum.ft2 centers.list --config dqfcosy.toml
```

A processed Bruker `pdata/<procno>` directory also works. Without window overrides,
PeakFit uses the acquisition and processing dictionaries loaded by nmrglue:
`acqu2s` and `proc2s` for the indirect axis, `acqus` and `procs` for the direct axis.
For supported Bruker processing, no configuration file is required:

```bash
uv run peakfit fit experiment/pdata/1 centers.list --lineshape dqfcosy
```

Automatic picking is unavailable for `dqfcosy` because the picker finds individual
lobes rather than multiplet centers. Existing models and defaults are unchanged.

## Override the detected processing variables

The following settings are examples. Replace them with the actual acquisition and
processing values for each axis. PeakFit labels the two spectral dimensions `F2`
and `F3`, reserving `F1` for the series dimension. `F3` is normally the direct axis.

```toml
[fitting]
lineshape = "dqfcosy"
dqfcosy_region_hz = 40.0

[fitting.dqfcosy_windows.F2]
acquisition_time = 0.064
points = 256
sine_power = 1
sine_offset = 0.1111111111
sine_end = 1.0
lb_hz = 0.0

[fitting.dqfcosy_windows.F3]
acquisition_time = 0.25
points = 2048
sine_power = 1
sine_offset = 0.0
sine_end = 1.0
lb_hz = 0.0
```

`points` is the number of retained **complex time-domain points**, before zero
filling. `acquisition_time` is that count divided by the acquisition sweep width
in Hz. The final sample occurs one dwell time before `acquisition_time`.
Neither variable is the processed spectrum size. For extracted frequency regions,
use the original acquisition sweep width to compute this time.
For NMRPipe data, if the header's Fourier-transform size differs from the current axis size,
PeakFit requires an explicit window override to avoid deriving acquisition timing
from the cropped sweep width.

Window settings are fixed inputs, not optimization parameters:

| Variable | Meaning |
| --- | --- |
| `sine_power = 0` | No sine weighting |
| `sine_power = 1` | Sine bell |
| `sine_power = 2` | Squared sine bell |
| `sine_offset`, `sine_end` | Start and end angles divided by π, each between 0 and 1 |
| `lb_hz` | Nonnegative exponential line broadening in Hz, optionally combined with a sine window |

For example, 20° corresponds to `sine_offset = 20/180`, entered as a decimal in
TOML. A cosine window uses offset 0.5 and end 1.0. No apodization uses power 0 and
LB 0. Exponential multiplication uses power 0 and the applied LB.

Without an explicit override for an axis, PeakFit reads NMRPipe APOD and APODQ
metadata for no window, SP with power 1 or 2, or EM with nonnegative LB. Verify
those headers against the processing script. A header may record only the last
window in a sequence. Explicit settings replace the entire detected window.
Unsupported windows fail with an error instead of falling back to a pseudo-Voigt.
Gaussian transformations, negative exponential broadening, partial or inverse
windows, linear prediction, magnitude processing, and nonuniform sampling are
outside this model.

For Bruker data, automatic detection supports `WDW=0` with no window, `WDW=1`
with exponential broadening from `LB`, `WDW=3` with SINE, and `WDW=4` with QSINE.
For SINE and QSINE, `SSB >= 2` gives a start angle of `π/SSB`; smaller nonnegative
values give an unshifted sine. Saved `LB` values do not apply to SINE or QSINE.

Timing comes from acquisition `TD` and `SW_h`, or `SW * SFO1`. `AQ_mod` identifies
the direct real/imaginary pairs. `FnMODE`, falling back to `MC2` when undefined,
identifies indirect quadrature. States, States-TPPI, and Echo-Antiecho use paired
samples. TPPI uses real samples at half the complex dwell interval. `TDeff` and
the Fourier-transform size limit the retained record; zero filling does not
increase its duration. Cropped processing sweep widths do not affect this timing.

The Bruker sine is defined over the acquisition interval. PeakFit translates its
angle at the last retained sample into `sine_end`, including when the time record
is truncated. The [TopSpin processing reference](https://nmr.chem.ucsb.edu/docs/Bruker_NMR_Manuals/processing-reference_v007.pdf)
defines these window and retention parameters in sections 2.4 and 3.27.

An explicit window block replaces detection for that axis only. Missing acquisition
metadata, unsupported `WDW`, linear prediction through `ME_mod`, or time-domain
shifts through `TDoff` produce an actionable DQF-COSY error. They do not prevent
loading the spectrum for other peak models.

Increase `dqfcosy_region_hz` until the rectangle includes all lobes and enough
surrounding signal to distinguish width from splitting. This is a half-width
around the marked center on each axis. Overlapping rectangles belong to the same
fit cluster. Check that neighboring unlisted peaks do not enter the region.

## Read the coupling result

For a peak named `A23`, the fitted coupling is `A23.F2-F3.j` in
`tables/parameters.csv` and `summary/fit.json`. Its unit is Hz. The two linewidths
are `A23.F2.lw` and `A23.F3.lw`, also in Hz. They describe exponential relaxation
before the specified window, with `R2 = π * lw` in s⁻¹. They are effective widths,
not a substitute for a dedicated relaxation measurement.

The initial J is 7 Hz, bounded between 0.1 and 20 Hz. Intrinsic linewidths start at
6 Hz and are bounded between 0.1 and 100 Hz. Change these through the existing
[parameter constraints](constraints-and-fit-steps.md), for example:

```toml
[parameters.peaks.A23]
"F2-F3.j" = { value = 4.5, min = 1.0, max = 12.0 }
```

Small phase errors can be fitted with `fit_phase = ["F2", "F3"]` under `[fitting]`.
These are cluster-level phase corrections, limited to ±15°. Start with a correctly
phased spectrum. Tilt fitting is rejected because a rotated product does not
describe this coupling geometry.

## Why lobe separation is not J

On each axis, PeakFit evaluates the difference between resonances centered at
`cs - J/2` and `cs + J/2`. Each resonance is the sampled transform of
`exp(-π * lw * t) * window(t)`. The two signed profiles multiply to produce the
four-lobe pattern, with opposite signs in adjacent quadrants. A fitted overall
amplitude accommodates a global sign reversal.

Finite acquisition, exponential relaxation, and the specified window all act
before the transform. Analytic geometric sums reproduce the finite sampled FID,
including the sine window's two endpoints. Zero filling changes where the model
is sampled, not the physical time record. The model includes destructive overlap
between opposite-sign components. Consequently, J is the component separation,
not the distance between the observed maxima. The signed multiplet integral is
zero; the reported fitted amplitude is not its volume or maximum lobe height.

The supplied `COSY.pdf`, section 6.2.1.5 and Figures 6.10–6.13, explains how
cancellation increases apparent splitting and weakens the cross-peak. Section
6.2.1.3 describes the effect of sine windows. The attachment's equation fonts are
damaged, including in rendered images; the implementation uses the explicit
transform above rather than transcribing the corrupted equations. See also
[Keeler's discussion of COSY and DQF-COSY](https://www-keeler.ch.cam.ac.uk/lectures/understanding/chapter_7.pdf)
and the [NMRPipe SP definition](https://www.nmrscience.com/ref/nmrpipe/sp.html).

## Assess whether J is determined

Inspect residuals, fitted bounds, and uncertainty before interpreting J. Try
several initial J values and reasonable fitting regions. Agreement across those
fits is more useful than optimizer convergence alone. Use the existing
[MCMC workflow](mcmc.md) to inspect J-width correlations; formal errors remain
conditional on the chosen model and noise assumptions. Zero-filled and apodized
spectra have correlated noise, so treating all processed points as independent can
underestimate uncertainty.

The model assumes weak coupling and equal component widths within each axis.
It does not include passive splittings, strong coupling, unequal multiplet
relaxation rates, exchange-specific lineshapes, or diagonal peaks. In particular,
do not interpret a four-lobe fit of a glycine multiplet as a reliable HN-Hα J.
Resolved Hα-side passive couplings also require a larger spin model.

The chapter cautions that J below roughly half the linewidth is difficult to
determine from antiphase splitting. A short time record can make J poorly
determined even at high signal-to-noise. Correct window modeling removes a source
of bias; it cannot recover information absent from the measurement.

`tests/test_dqfcosy.py` compares the model against independently transformed FIDs,
checks the shared-J Jacobian, protects the center-based fitting region, and fits
noisy synthetic cross-peaks with separate relaxation and window settings. These
checks establish implementation correctness, not validation on experimental data.
