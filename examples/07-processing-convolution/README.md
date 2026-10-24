# Frequency-domain convolution benchmark

This experiment implements the proposed frequency-domain convolution for tilted
pseudo-Voigt peaks and compares it with the production window-aware evaluator.
It did **not** improve fitting time for the tested GB1 peak. The production
`--include-window-functions` path remains unchanged.

## Reproduce the comparison

Run from the repository root. The selected peak must be the only listed peak
in its contour-connected cluster. The benchmark expects a processed Bruker
2D spectrum with an adjacent `peaklist.xml`.

```bash
OPENBLAS_NUM_THREADS=1 uv run examples/07-processing-convolution/benchmark.py \
    /path/to/experiment/pdata/1/2rr --peak peak2 --output /tmp/convolution.json
uv run examples/07-processing-convolution/verify.py
```

Both fits start with intrinsic linewidths of 51 Hz and 100 Hz, mixing
coefficients of 0.5, and tilt of 20 degrees. They share the peak positions,
contour mask, experimental intensities, noise estimate, and optimizer settings.
This tests one isolated cluster, rather than fitting the entire dataset.
The input data is not modified.

## Implementation

`ConvolutionCluster` samples the intrinsic tilted pseudo-Voigt profile in
frequency space. It precomputes the Fourier transform of each recorded window,
including truncation and first-point scaling. Each axis uses FFT-based circular
convolution with normalization by its lattice length.

The convolution spans the complete acquisition frequency period. Its spacing
matches the stored spectral spacing, so output values are selected directly
without interpolation. The lattice is aligned near the peak while preserving
the stored spectral coordinates. The acquisition dwell determines the frequency
period even when the stored spectrum is cropped.

The implementation batches the intrinsic values and analytic derivatives. It
works in blocks of proton frequencies, convolves the nitrogen axis, and keeps
only the requested nitrogen rows before convolving the proton axis. This avoids
retaining all full-plane derivative arrays.

The experiment covers real 2D pseudo-Voigt singlets without fitted phases.
It does not replace the broader production model. Sampling the intrinsic profile
introduces quadrature and tail errors that depend on widths, tilt, and grid
spacing. Agreement at one set of parameters is not a universal error bound.
Oversampling is available through `ConvolutionCluster.prepare` for numerical
checks, but increases computation and memory use.

## Measured result

The saved [benchmark-results.json](benchmark-results.json) records the GB1
`peak2` comparison with one BLAS thread. Hardware and other running jobs affect
absolute timings.

| Model | Complete one-cluster fit | Evaluations | Reduced chi-squared |
| --- | ---: | ---: | ---: |
| Direct sampled time signal | 5.96 s | 9 | 0.4677720 |
| Blocked frequency convolution | 28.14 s | 9 | 0.4677741 |

Both fits converged. Convolution was about **4.7 times slower** in this run.
The initial model differed from the direct reference by about 1.1 parts per
million of the reference peak height. The largest relative derivative error
was about 0.005%. The fitted widths differed by less than 0.0012 Hz, the tilt by
less than 0.00009 degrees, and the positions by less than 0.00000021 ppm.

The convolution lattice has 1024 by 8192 samples, about 8.4 million values per
peak. The production time-domain calculation has 143 by 3987 samples, about
570,000 values. The small cropped spectrum does not remove the convolution's
need to account for surrounding intrinsic signal and the acquisition period.
Generating and transforming that larger frequency lattice outweighed the benefit
of cached kernels and blocked transforms for this cluster.

Smaller local grids were also tried in a temporary prototype. They reduced the
frequency extent and resampled the windows, but produced relative model errors
of roughly 0.8–3%. Those approximations were not adopted.

## Verification

`verify.py` checks FFT convolution against explicit sums of shifted arrays,
checks all seven analytic derivatives against finite differences, compares
pure-Gaussian input with the independent production forward model, and checks
a noncentral crop against the corresponding full-spectrum region. The saved
GB1 comparison separately checks mixed Gaussian/Lorentzian profiles against
the production values, derivatives, and fitted results.
