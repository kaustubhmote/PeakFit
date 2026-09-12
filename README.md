# PeakFit

PeakFit is a Python package and CLI for fitting lineshape models to pseudo-ND NMR
spectra.

The normal workflow is:

```bash
peakfit fit spectrum.ft2 peaks.list
peakfit mcmc Fits/<run_dir>
peakfit plot intensity Fits/<run_dir>
peakfit plot cest Fits/<run_dir>
peakfit plot spectrum --spectrum spectrum.ft2 --results Fits/<run_dir>
```

Start with [`examples/02-advanced-fitting`](examples/02-advanced-fitting/) for a
ready-to-run pseudo-ND CEST workflow.

## Install

PeakFit requires Python 3.14 or newer.

```bash
git clone https://github.com/gbouvignies/PeakFit.git
cd PeakFit
uv sync --all-extras
uv run peakfit --help
```

For an installed command in the current environment:

```bash
uv pip install .
peakfit --help
```

## Fit

Use an NMRPipe spectrum (`.ft2` or `.ft3`) or a processed Bruker spectrum and a peak list:

```bash
peakfit fit data/pseudo3d.ft2 data/pseudo3d.list \
  --z-values data/b1_offsets.txt \
  --output Fits
```

For Bruker data, pass any of these paths:

```bash
peakfit fit /data/sample/10/pdata/1 peaks.list
peakfit fit /data/sample/10/pdata/1/2rr peaks.list
peakfit fit /data/sample/10 peaks.list
```

The last form selects `pdata/1`. If an experiment has several processed datasets
and no `pdata/1`, select the required `pdata/<procno>` directory. PeakFit reads
processed `1r`, `2rr`, `3rrr`, and `4rrrr` data; it does not process raw Bruker
`fid` or `ser` data.

`peakfit fit` validates inputs before fitting. Omitting the peak list invokes the
experimental automatic peak-picking workflow; provide a peak list for routine
analyses.

Useful fit options:

- `--config peakfit.toml` for reproducible settings.
- `--lineshape auto|gaussian|lorentzian|pvoigt|sp1|sp2|no_apod`.
- `--tilted` to fit a rotated peak in a two-dimensional spectral plane.
- `--optimizer varpro|basin_hopping`.
- `--refine N` for exactly `N` optimizer passes; corrections update only between
  passes. Existing commands that relied on the former `N + 1` pass behavior
  should increment `N` by one.
- `--workers -1` to use all CPUs.
- `--format json --format csv --format txt` to request output formats.

Generate a starter config:

```bash
peakfit init peakfit.toml
```

### Fit tilted two-dimensional peaks

Use `--tilted` with any existing lineshape:

```bash
peakfit fit /data/sample/10/pdata/1/2rr peaks.list \
  --lineshape pvoigt \
  --tilted
```

The equivalent configuration is:

```toml
[fitting]
lineshape = "pvoigt"
tilted = true
```

Tilt is independent of the one-dimensional profile. It works with `gaussian`,
`lorentzian`, `pvoigt`, `sp1`, `sp2`, and `no_apod`, including each
`_doublet` variant. `--lineshape auto --tilted` also works.

Each peak gets a fitted `<peak>.F2-F3.tilt` parameter in degrees. The angle is
measured counterclockwise in the F2-F3 ppm plane. You can constrain it with the
normal parameter rules:

```toml
[parameters.defaults]
"*.*.tilt" = { min = -60.0, max = 60.0 }

[parameters.peaks."G45N-HN"]
"F2-F3.tilt" = { value = 35.0 }
```

Tilted fitting requires exactly two spectral dimensions. The angle is poorly
determined for a nearly circular peak or a heavily overlapped peak. Compare an
axis-aligned run and a tilted run with the reported AIC or BIC before accepting
the extra parameter.

SP1, SP2, and no-apodization tilt the complete processed profile. Treat these
as phenomenological models. The `_doublet` variants rotate the complete
doublet surface, including its splitting pattern.

## Inputs

Peak lists can be:

- Sparky-style `.list` files with `Assignment w1 w2 ...` columns.
- Tables (`.csv`, `.json`, `.xlsx`, `.xls`) with position columns such as
  `F1_ppm`, `w1`, or `Pos F1`.

For pseudo-ND data, use the retained `--z-values` option to supply one plane
value per plane. CEST examples use B1 offset values; CPMG plotting additionally
needs `--time-t2`.

For Bruker spectra, `--lineshape auto` uses `no_apod` when `WDW=0` and the
robust `pvoigt` model for other window functions. Select an explicit lineshape
when you need a processing-specific model.

## Outputs

By default, `peakfit fit --output Fits` writes a timestamped run directory:

```text
Fits/<run_dir>/
├── README.md
├── summary/
│   └── fit.json
├── tables/
│   ├── parameters.csv
│   ├── intensities.csv
│   └── shifts.csv
├── metadata/
│   └── fitting_state.pkl
```

The canonical machine-readable summary is `summary/fit.json`. The canonical
tables are CSV files under `tables/`. Markdown reports are optional and are
written only when `txt` is requested.

See [`docs/outputs.md`](docs/outputs.md) for the exact output
contract.

## Plot

```bash
LATEST_RUN="$(find Fits -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"

peakfit plot intensity "$LATEST_RUN" --output "$LATEST_RUN/intensity_profiles.pdf"
peakfit plot cest "$LATEST_RUN" --output "$LATEST_RUN/cest_profiles.pdf"
peakfit plot cpmg "$LATEST_RUN" --time-t2 0.04 --output "$LATEST_RUN/cpmg_profiles.pdf"
peakfit plot spectrum --spectrum data/pseudo3d.ft2 --results "$LATEST_RUN"
peakfit plot spectrum --spectrum /data/sample/10/pdata/1/2rr --results "$LATEST_RUN"
```

The spectrum viewer accepts the same NMRPipe files and processed Bruker paths as
`peakfit fit`.

CEST and CPMG plots are generated from `tables/intensities.csv`. CEST preserves
signed normalized intensities. CPMG computes `R2eff` from `log(I/I0)`, so only
points with a positive intensity ratio can contribute to the transformed profile.

## MCMC

Run MCMC after a successful fit:

```bash
peakfit mcmc "$LATEST_RUN" --peaks 2N-H --walkers 32 --steps 1000
peakfit plot mcmc "$LATEST_RUN" --output "$LATEST_RUN/mcmc_diagnostics.pdf"
```

`peakfit mcmc` reads `metadata/fitting_state.pkl` and writes chain files under
`chains/` when `--save-chains` is enabled.

## Documentation

- [`docs/outputs.md`](docs/outputs.md) - output files and formats.
- [`docs/constraints-and-fit-steps.md`](docs/constraints-and-fit-steps.md) -
  parameter constraints and `[[fitting.steps]]`.
- [`docs/optimizers.md`](docs/optimizers.md) - optimizer choice and runtime controls.
- [`docs/mcmc.md`](docs/mcmc.md) - MCMC workflow and diagnostics.
- [`docs/development.md`](docs/development.md) - contributor workflow and coding conventions.
- [`docs/architecture.md`](docs/architecture.md) - current package map and boundaries.
- [`AGENTS.md`](AGENTS.md) - instructions for AI-assisted maintenance.

## Development

```bash
uv sync --all-extras
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check --error-on-warning
uv run lint-imports
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg uv run pytest -q
uv run prek run --all-files
uv build
```

Use `uv` for project commands. Use `prek`, not `pre-commit`, for hooks.

## License

GPL-3.0-or-later.
