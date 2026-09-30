# Customize spectrum contours

Pass a TOML file to the spectrum viewer:

```bash
peakfit plot spectrum --spectrum data/pseudo3d.ft2 \
  --results Fits/<run_dir> --config peakfit.toml
```

The file can contain only plotting settings, or the same fitting configuration
you use with `peakfit fit`. The viewer reads its `[plot]` section. Without
`--config`, it uses the existing appearance. It does not automatically discover
a configuration file or load the fitting run's configuration.

Edit each spectrum layer independently:

```toml
[plot.experimental]
color = "#1f77b4"
linewidth = 0.75

[plot.simulated]
color = "#ff7f0e"
linewidth = 1.0

[plot.difference]
colormap = "RdBu_r"
linewidth = 0.5
```

- `color` accepts a Matplotlib color name, a hex code, or a cycle color such as
  `C0`. Both positive and negative contours use this fixed color.
- `linewidth` sets the contour width in points. It must be finite and positive.
  Omit it to use Matplotlib's default, normally 1.5 points.
- `colormap` accepts a Matplotlib colormap name, such as `Blues`, `Oranges`, or
  `RdBu_r`. It colors contours by their signed level and takes precedence over
  `color`. Remove `colormap` to return to a fixed color. Each layer uses its own
  colormap with the same symmetric contour levels.

Omitted layers and fields retain their defaults. The default colors are `C0`
for experimental, `C1` for simulated, and `C2` for difference. Contour opacity
remains 0.7, and contour thresholds and spacing retain their existing behavior.
Invalid colors, colormaps, widths, and unknown configuration keys produce errors.

Restart the viewer after editing the file. Enable **Diff** to inspect the
difference spectrum. Simulated and difference spectra require `--results`.
Use the contour control to adjust the threshold independently of the style.

`peakfit init peakfit.toml` includes editable plotting sections. These settings
apply to the spectrum viewer, not the intensity, CEST, CPMG, or MCMC PDFs.
Matplotlib script export is not implemented yet.

The advanced fitting example includes a [starter file](../examples/02-advanced-fitting/data/plot.toml)
with the current colors and commented colormaps.
