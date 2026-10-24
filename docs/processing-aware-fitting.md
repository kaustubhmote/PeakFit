# Fit intrinsic peaks with recorded windows

Use `--include-window-functions` when you want the fitted linewidths to describe
profiles before apodization. PeakFit reads the window and acquisition sampling
from NMRPipe headers or Bruker acquisition and processing status files.

```bash
uv run peakfit fit spectrum.ft2 peaks.list --include-window-functions --lineshape lorentzian
uv run peakfit fit experiment/pdata/1/2rr --include-window-functions --tilted --headless
```

The same option is available in TOML:

```toml
[fitting]
include_window_functions = true
lineshape = "pvoigt"
tilted = true
```

The CLI flag enables window functions even when a TOML file is supplied.
Without the flag or TOML setting, fitting behavior is unchanged.

## Intrinsic profile and processing

With this option, `auto` selects intrinsic pseudo-Voigt profiles. You can instead
select Gaussian or Lorentzian profiles. Their doublet variants are supported.
Existing `no_apod`, `sp1`, and `sp2` shapes already contain processing assumptions
and cannot be combined with this option. DQF-COSY retains its existing
window-aware model and per-axis overrides.

PeakFit Fourier-transforms the intrinsic profile into a time signal, samples it
at the acquisition dwell interval, multiplies it by the recorded window, and
transforms it back at the available spectral coordinates. The finite sample
count includes truncation effects. Zero filling supplies spectral coordinates
without extending acquisition time. First-point scaling is included.

For tilted 2D peaks, PeakFit rotates the intrinsic profile in ppm space and
applies each window along its acquisition axis. Rotating two already-windowed
1D profiles would apply a different processing operation. Analytic derivatives
cover positions, widths, mixing coefficients, doublet couplings, tilt, and
optional fitted zero-order phases. As in existing fits, phase parameters are
shared within a cluster. The input is assumed to be phased for absorption.
Header phase corrections are not applied a second time.

The model evaluates one or two spectral dimensions. Imaginary component fitting
is available through the existing DQF-COSY model only. Window-aware transforms
cost more than empirical profiles, especially for tilted peaks with long FIDs.

## Supported windows

| Input | Recorded window | Parameters |
| --- | --- | --- |
| NMRPipe | None, `APODCODE=0` | `APOD`, acquisition sweep width |
| NMRPipe | Sine bell, `APODCODE=1` | `APODQ1`, `APODQ2`, `APODQ3` |
| NMRPipe | Exponential, `APODCODE=2` | `APODQ1` in Hz |
| NMRPipe | Lorentz-to-Gauss, `APODCODE=3` | `APODQ1`, `APODQ2` in Hz, `APODQ3` center |
| NMRPipe | Modified Gaussian, `APODCODE=7` | `APODQ1` in Hz, `APODQ2` fraction |
| Bruker | None, `WDW=0` | Acquisition sampling |
| Bruker | Exponential, `WDW=1` | `LB` in Hz |
| Bruker | Gaussian multiplication, `WDW=2` | Negative `LB`, `GB` fraction |
| Bruker | Sine or squared sine, `WDW=3` or `4` | `SSB` |

NMRPipe `C1` stores the first-point multiplier minus one. Bruker `FCOR` is the
multiplier itself. Window codes from the two formats are not interchangeable.
Unknown windows and incomplete sampling metadata produce an error when the
option is enabled. Ordinary spectrum loading remains available.

The NMRPipe equations follow the official
[SP](https://spin.niddk.nih.gov/NMRPipe/ref/nmrpipe/sp.html),
[GM](https://spin.niddk.nih.gov/NMRPipe/ref/nmrpipe/gm.html), and
[GMB](https://spin.niddk.nih.gov/NMRPipe/ref/nmrpipe/gmb.html) references.
Bruker Gaussian multiplication uses `a=pi*LB`, `b=-a/(2*GB*AQ)`, and
`exp(-a*t-b*t*t)`, with acquisition duration `AQ`. The definitions of `WDW`,
`SSB`, and `FCOR` follow the
[Bruker processing reference](https://nmr.chem.ucsb.edu/docs/Bruker_NMR_Manuals/processing-reference_v007.pdf).

## Extracted spectra and acquisition timing

For Bruker, PeakFit uses `SW_h`, or `SW*SFO1`, from `acqus` and `acqu2s`, with
quadrature pairing from `AQ_mod` and `FnMODE` or `MC2`. It uses `TD` for the
acquired sample count and `TDeff` for retained samples. Processed `SW_p`, `STSI`,
and the displayed array length do not determine the dwell interval. A shortened
FID retains the original duration that defines its Gaussian or sine window.

For a standard NMRPipe frequency extraction with `EXT -sw`, PeakFit uses `X1`,
`XN`, and `FTSIZE` to recover the sweep width and window sample count before
extraction. Displayed ppm coordinates retain their extracted calibration.
See the [NMRPipe EXT reference](https://spin.niddk.nih.gov/NMRPipe/ref/nmrpipe/ext.html).
Repeated extractions, manual header changes, or extraction without consistent
calibration metadata are not reconstructed. Integer rounding of `APOD` during
extraction can lose the exact original sample count. Use an unextracted file
when that rounding matters. Headers also do not describe a complete processing
history, so composite or inverted windows cannot be inferred reliably.

## Interpretation and limits

Linear prediction is deliberately ignored. Bruker predicted points do not
extend the modeled FID. A predicted experimental spectrum can therefore retain
residual differences, even with the correct window on acquired samples.
Time-domain shifts are unsupported. Solvent filtering, baseline correction,
resampling, and arbitrary processing histories are outside this model.

A fitted `lw` is the intrinsic FWHM under the selected profile. For tilted peaks,
it is the width on the corresponding principal axis. Gaussian disorder,
unresolved components, or an inadequate profile can still bias that width.
Including the window does not identify the physical origin of tilt. Signed
pseudo-Voigt coefficients remain empirical shape parameters, rather than
population fractions. Windowing and zero filling also correlate noise, so
ordinary covariance errors remain conditional on the noise model.

Start a new fit when enabling the option. Previous widths and amplitudes were
estimated with a different forward model. Processing-aware amplitudes multiply
unit-height intrinsic profiles, and need not equal the observed processed peak
height. Do not compare those amplitudes directly with empirical-fit heights.

## Verification

`tests/test_processing_aware.py` compares recorded NMRPipe windows with nmrglue,
checks complex-FID transforms and first-point factors, compares a tilted
Gaussian with an independent frequency-domain FFT and windowing calculation,
and verifies full-spectrum versus extracted-spectrum equivalence. It also
checks analytic derivatives and recovery of intrinsic widths and tilt through
the optimizer. Existing tilted and DQF-COSY regression tests protect the
unchanged fitting paths.

## Frequency convolution comparison

A frequency-domain FFT convolution implementation was tested on an isolated GB1
peak. It reproduced the fitted result closely, but took about 4.7 times longer
than the direct model in that run. The acquisition-period frequency lattice
was much larger than the direct time grid, despite the cropped output region.
The production evaluator remains unchanged. See the reproducible
[convolution benchmark](../examples/07-processing-convolution/README.md).
