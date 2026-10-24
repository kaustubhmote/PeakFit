"""Fit a supplied processed file; no datasets are downloaded or bundled.

Example:
uv run examples/python-api/fit_file_region.py spectrum.ucsf --direct-axis 2 \
    --region 0:50:65 --region 1:110:130 --region 2:7:9 --noise 0.1

Every frequency axis requires a region. These ppm values are illustrative.
Picking is explicitly experimental; use PeakSeed records for assigned peaks.
"""

import argparse

from peakfit.api import fit_file


def main():
    """Read native coordinates and run the same regional preparation as arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum")
    parser.add_argument("--direct-axis", type=int)
    parser.add_argument("--region", action="append", required=True)
    parser.add_argument("--noise", type=float, required=True)
    args = parser.parse_args()
    bounds = {}
    for item in args.region:
        axis, low, high = item.split(":")
        if int(axis) in bounds:
            parser.error(f"Duplicate region for external axis {axis}")
        bounds[int(axis)] = (float(low), float(high))
    result = fit_file(
        args.spectrum,
        direct_axis=args.direct_axis,
        region_ppm=bounds,
        noise=args.noise,
        auto_pick=True,
    )
    for peak in result.peaks:
        print(peak.id, dict(peak.positions_ppm), peak.amplitudes, peak.classification)  # noqa: T201


if __name__ == "__main__":
    main()
