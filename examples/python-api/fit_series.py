"""Run with uv run examples/python-api/fit_series.py from the repository root."""

import numpy as np
from fit_array import example_input

from peakfit.api import PeakFitConfig, PeakSeed, SeriesAxis, SpectrumInput, fit_spectrum


def main():
    """Fit shared 3D positions/widths and ordered, independent series amplitudes."""
    data, udic, positions, region = example_input()
    ratios = (1.0, 0.5, 1.8, -0.3)
    # Deliberately put series at external axis 1, between frequency axes.
    series_data = np.stack([data * ratio for ratio in ratios], axis=1)
    mapping = {0: 0, 1: 2, 2: 3}
    series_udic = {"ndim": 4, **{mapping[i]: udic[i] for i in range(3)}}
    series_udic[1] = {
        "size": 4,
        "sw": 1.0,
        "obs": 1.0,
        "car": 0.0,
        "label": "relaxation",
        "freq": False,
        "time": True,
        "complex": False,
    }
    series = SeriesAxis((20, 5, 20, 10), "ms", "relaxation", ("a", "b", "replicate-a", "c"))
    source = SpectrumInput.from_array(
        series_data, series_udic, direct_axis=3, series_axis=1, series=series
    )
    config = PeakFitConfig(noise_level=0.1)
    config.fitting.lineshape = "gaussian"
    result = fit_spectrum(
        source,
        config=config,
        peaks=[PeakSeed("shared-peak", {mapping[k]: v for k, v in positions.items()})],
        region_ppm={mapping[k]: v for k, v in region.items()},
    )
    print(result.extraction.series)  # noqa: T201
    print(result.peaks[0].amplitudes)  # noqa: T201


if __name__ == "__main__":
    main()
