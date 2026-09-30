# Tilted fitting of the GB1 hNH spectrum

The best supported empirical model is a tilted pseudo-Voigt profile with
negative mixing allowed and a 3-noise segmentation contour. It fits all 57
supplied peaks in 38 clusters. Every optimizer converged, but the residual
contains systematic shoulders and side lobes. This is a useful descriptive fit,
not a validated model of intrinsic linewidths.

## Data and processing

The input is `experimental-55.555kHz-CDN/01-Assignments/1/pdata/1/2rr` and its
adjacent TopSpin `peaklist.xml`. The pulse-program header identifies `hNH_v2.krm`.
The spectrum has 1024 nitrogen points and 256 proton points. The list contains
57 positions across four XML groups, with three negative peaks. PeakFit names
them `peak1` through `peak57` in XML order, without residue assignments.

The robust noise estimate is 5522.64 intensity units. Proton digital spacing is
0.017433 ppm, or 12.207 Hz. Nitrogen spacing is 0.024772 ppm, or 1.758 Hz.
Interpolation and zero filling do not supply independent observations.

Both axes use Gaussian multiplication, `WDW=2`, with `GB=0.3`. The proton axis
has `LB=-50 Hz`, and the nitrogen axis has `LB=-10 Hz`. Bruker documents negative
LB for Gaussian multiplication in its
[processing reference](https://nmr.chem.ucsb.edu/docs/Bruker_NMR_Manuals/processing-reference_v007.pdf).
These settings sharpen the spectrum and make an everywhere-positive profile
an incomplete description of some observed tails.

The nitrogen status header also records `ME_mod=2`, `LPBIN=288`, and `TDeff=288`,
whereas the acquired `TD` is 144. This is a separate processing effect that an
exact model must address. It must not silently treat predicted samples as
additional experimental observations.

## Comparisons

The first five runs use exactly the same 14,545 observations, 43 clusters,
position bounds, and initialization. The script checks equality of their
observation masks. Values below are rounded, and RMS uses the original spectrum
intensity units.

| Model | RMS in the common fitting mask | Full-spectrum RMS | Reduced chi-squared | BIC |
| --- | ---: | ---: | ---: | ---: |
| Axis-aligned Gaussian | 44,500 | 13,078 | 66.23 | 1,224,502 |
| Tilted Gaussian | 25,448 | 9,108 | 21.74 | 589,501 |
| Tilted pseudo-Voigt, eta from 0 to 1 | 23,195 | 9,090 | 18.21 | 538,296 |
| Axis-aligned pseudo-Voigt, eta from -1 to 1 | 41,541 | 13,542 | 58.17 | 1,104,127 |
| Tilted pseudo-Voigt, eta from -1 to 1 | 22,475 | 9,027 | 17.10 | 522,639 |
| Recommended broader-region fit | 22,708 | 8,853 | 15.73 | Different observations |

Tilt lowers the Gaussian fitting-region squared residual by 67.3%. The signed
pseudo-Voigt lowers it by 74.5% relative to the axis-aligned Gaussian.
The recommended fit lowers full-spectrum RMS by 32.3% relative to that baseline.

The recommended run uses 19,664 observations, five passes, and a wider angle
range. Its BIC cannot be compared directly with the five baseline runs.
Its slightly larger common-mask RMS and smaller full-spectrum RMS show the
tradeoff between fitting peak cores and their tails. A contour-sensitivity run
at 8-noise gave a full-spectrum RMS of 9396, worse than the 3-noise run.

The processed data are correlated, and the fit masks depend on observed signal.
Reported AIC, BIC, and chi-squared are descriptive comparisons here. They do not
establish calibrated statistical confidence.

## Recommended constraints

`recommended.toml` contains the following settings:

- Position windows of +/-0.20 ppm for nitrogen and +/-0.045 ppm for proton.
- Principal-profile width bounds of 5 to 120 Hz for nitrogen and 12 to 250 Hz for proton.
- Tilt bounds of -60 to 60 degrees in the internal F2-F3 ppm coordinate system.
- Independent mixing parameters from -1 to 1 on each principal profile.
- Five refinement passes, with segmentation at three times the noise estimate.

Negative eta subtracts a Lorentzian component from the Gaussian component.
It is an empirical way to represent sharpening side lobes, not a physical
Lorentzian fraction. The resulting widths describe the processed principal
profiles and cannot be interpreted as relaxation rates.

The recommended median tilt is 12.63 degrees, with an interquartile range of
8.82 to 15.11 degrees. Median principal-profile widths are 19.02 Hz for nitrogen
and 75.96 Hz for proton. Widths and angles need particular care for nearly
circular or overlapping peaks because those parameters can trade against one another.

The fitted negative heights belong to `peak52`, `peak53`, and `peak54`, matching
the negative entries in the XML. No extra peaks or doublet splittings were added.

## Remaining mismatch and useful next changes

The largest residual costs occur around `peak37` with `peak50`, `peak49`, the
`peak18`/`peak28`/`peak51`/`peak56` overlap, and `peak41`. Isolated strong peaks
also show repeated diagonal shoulders. That repetition makes additional
resonances a poor default explanation. The plots show where the hypothesis
of one symmetric processed peak fails.

Several eta values reach their bounds, including the nitrogen values of
`peak12`, `peak18`, and `peak57`. Increasing optimizer passes or tightening
positions cannot resolve a missing profile feature. Formal covariance errors
also understate uncertainty when the model has systematic residuals.

The most useful next model is a tilted intrinsic peak with processing applied
along the original acquisition axes. In the time domain, it would combine
intrinsic broadening and correlated frequency variation with the actual GM
windows, sampling, and truncation. In the frequency domain, that is convolution
with axis-specific window responses. Rotating the complete processed profile,
as the present empirical model does, rotates the window response too.

The model must use acquisition timing rather than the width of the cropped
spectrum. For proton, `TD=3988` real values at `SW_h=100000 Hz` give 1994 complex
points and an acquisition duration of 0.01994 s. The old generic Bruker reader
used the extracted 3125-Hz width and returned 0.63808 s. This branch corrects
that calculation while preserving the cropped ppm coordinates, explicit `AQ`,
and the fallback for incomplete acquisition metadata. Pseudo-Voigt results are
unchanged because that model does not use acquisition duration.

For nitrogen, fitting the originally acquired FID would avoid modeling the
linear-prediction transformation. A separate processing experiment with less
sharpening and no linear prediction would help determine whether the repeated
shoulders persist. Reprocessing is a proposed validation step, not a result of
this analysis.

An exploratory absorptive/dispersive extension reduced core residuals, but
increased full-spectrum RMS to 10,742 or 12,453 when additional skew terms were
allowed. Those unconstrained phenomenological extensions were rejected.
A processing-aware model needs tail validation rather than selection on peak
cores alone. These exploratory trials are separate from the supported-model
comparison script.

## Verification

The comparison script reproduced all six runs with finite full-grid models,
equal baseline masks, and 57 usable fitted peaks. Diagnostic figures were
rendered and inspected at identical positive and negative contour levels.
The source timing change has regression coverage for cropped direct and
indirect axes, both acquisition-width encodings, explicit `AQ`, and the
incomplete-metadata fallback. The focused suite passed 112 tests, and changed
Python files passed Ruff, formatting, and type checks.

No new lineshape or CLI option was introduced. The only library behavior change
is corrected acquisition timing when acquisition and extracted sweep widths
differ. Existing fits with duration-dependent profiles should be rerun for
such spectra. Generated results remain outside Git, and the source data were
not modified.
