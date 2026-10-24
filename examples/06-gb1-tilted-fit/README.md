# Fit the GB1 Bruker spectrum with tilted profiles

This example compares five profiles on identical observations, then fits a
broader region with the recommended empirical settings. The experimental data
remain external to the repository. See [findings.md](findings.md) for the
measurements, limitations, and proposed processing-aware model.

From the repository root, run:

```bash
uv run examples/06-gb1-tilted-fit/fit_dataset.py \
  /home/kaustubh/data/research/gb1/experimental-55.555kHz-CDN/01-Assignments/1/pdata/1/2rr
```

The script reads `peaklist.xml` beside the spectrum and writes six runs under
`results/`. It uses one worker and a noninteractive plotting backend.

Inspect these generated files:

- `results/fit-overview.png`: experiment, recommended model, and residual at identical contour levels.
- `results/worst-residuals.png`: the twelve regions with the largest residual cost.
- `results/comparison.csv`: full-spectrum and common-region RMS, observation masks, and fit statistics.
- `results/recommended/tables/parameters.csv`: fitted positions, widths, mixing parameters, and tilt.
- `results/recommended/tables/intensities.csv`: signed fitted peak heights.
- `results/recommended/model.npz`: model, residual, ppm axes, noise, and fitting mask.
- `results/recommended/summary/fit.json`: canonical scientific results and input checksums.

For one CLI fit with the same recommended settings, run:

```bash
uv run peakfit fit \
  /home/kaustubh/data/research/gb1/experimental-55.555kHz-CDN/01-Assignments/1/pdata/1/2rr \
  --config examples/06-gb1-tilted-fit/recommended.toml --workers 1
```

The CLI writes the canonical tables and state. Use the comparison script to
produce the full-grid NPZ and diagnostic plots. The NPZ preserves the cropped
Bruker coordinates without converting the processing dictionary to NMRPipe.

Generated results are ignored by Git. `recommended.toml` contains no warm-start
dependency, so it can reproduce the fit directly from the spectrum and peak list.
