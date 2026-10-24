"""Run with uv run examples/python-api/fit_array.py from the repository root."""

import numpy as np

from peakfit.api import PeakFitConfig, PeakSeed, SpectrumInput, fit_spectrum


def example_input():
    """Return an independently generated, referenced true-3D Gaussian spectrum."""
    shape = (21, 25, 29)
    udic = {"ndim": 3}
    data = np.ones(shape, dtype=np.float64)
    positions = {}
    region = {}
    for axis, (size, obs, center, label) in enumerate(
        zip(shape, (150.0, 60.0, 600.0), (55.0, 120.0, 8.0), ("13C", "15N", "1H"), strict=True)
    ):
        sw = 240.0
        grid = center + sw / (2 * obs) - np.arange(size) * sw / (size * obs)
        positions[axis] = center + 0.1 * sw / (size * obs)
        udic[axis] = {
            "size": size,
            "sw": sw,
            "obs": obs,
            "car": center * obs,
            "label": label,
            "freq": True,
            "time": False,
            "complex": False,
        }
        profile = np.exp(-4 * np.log(2) * ((grid - positions[axis]) * obs / 25) ** 2)
        broadcast = [1, 1, 1]
        broadcast[axis] = size
        data *= profile.reshape(broadcast)
        region[axis] = (float(grid[2]), float(grid[-3]))
    return 100 * data, udic, positions, region


def main():
    """Fit a small region with known coordinates, returning no files."""
    data, udic, positions, region = example_input()
    config = PeakFitConfig(noise_level=0.1)
    config.fitting.lineshape = "gaussian"
    source = SpectrumInput.from_array(data, udic, direct_axis=2)
    result = fit_spectrum(
        source, peaks=[PeakSeed("assigned-peak", positions)], region_ppm=region, config=config
    )
    for peak in result.peaks:
        print(peak.id, dict(peak.positions_ppm), peak.amplitudes, peak.classification)  # noqa: T201


if __name__ == "__main__":
    main()
