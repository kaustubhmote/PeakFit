"""Compare constrained tilted profiles on a supplied Bruker 2D spectrum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from peakfit.engine.domain.constraints import ParameterConstraint
from peakfit.fit.fitting import load_data, run_fit, write_fit_run_outputs
from peakfit.fit.simulation import simulate_final_outcome
from peakfit.io.config import load_config, save_config
from peakfit.io.readers.spectrum import resolve_spectrum_path

CASES = (
    ("gaussian-axis", "gaussian", False, 0),
    ("gaussian-tilt", "gaussian", True, 0),
    ("pvoigt-tilt", "pvoigt", True, 0),
    ("pvoigt-signed-axis", "pvoigt", False, -1),
    ("pvoigt-signed-tilt", "pvoigt", True, -1),
    ("recommended", "pvoigt", True, -1),
)


def draw_contours(ax, x, y, values, levels):
    """Use the same positive and negative levels in every panel."""
    ax.contour(x, y, values, levels=levels, colors="#24549a", linewidths=0.65)
    ax.contour(x, y, values, levels=-levels[::-1], colors="#d05232", linewidths=0.65)
    ax.set_xlabel("1H / ppm")
    ax.set_ylabel("15N / ppm")
    ax.invert_xaxis()
    ax.invert_yaxis()


def plot_results(output, loaded, fitted, model):
    """Plot the full spectrum and the twelve regions with the largest residual cost."""
    data = loaded.spectra.data[0]
    y = loaded.spectra.params[1].pts2ppm(np.arange(data.shape[0]))
    x = loaded.spectra.params[2].pts2ppm(np.arange(data.shape[1]))
    residual = data - model[0]
    levels = loaded.noise * 5 * 1.5 ** np.arange(10)
    fig, axes = plt.subplots(1, 3, figsize=(18, 11), layout="constrained")
    titles = ("Experimental", "Tilted signed pseudo-Voigt", "Residual, same contour levels")
    for ax, values, title in zip(axes, (data, model[0], residual), titles, strict=True):
        draw_contours(ax, x, y, values, levels)
        ax.set_title(title)
    fig.savefig(output / "fit-overview.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 12), layout="constrained")
    draw_contours(ax, x, y, data, levels)
    for peak in loaded.peaks:
        nitrogen, proton = peak.positions
        ax.plot(proton, nitrogen, "k+", ms=4)
        ax.annotate(
            peak.name, (proton, nitrogen), xytext=(3, 3), textcoords="offset points", fontsize=6
        )
    ax.set_title("GB1: supplied peak list, blue positive / orange negative")
    fig.savefig(output / "spectrum.png", dpi=150)
    plt.close(fig)

    ranked = sorted(
        fitted.outcome.clusters,
        key=lambda cluster: cluster.analytical_evaluation.statistics.chi_squared,
        reverse=True,
    )
    clusters = {cluster.cluster_id: cluster for cluster in loaded.clusters}
    parameters = {p.name: p.value for p in fitted.outcome.final_nonlinear_parameters}
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), layout="constrained")
    for ax, outcome in zip(axes.ravel(), ranked[:12], strict=False):
        coordinates = np.array([p.positions for p in clusters[outcome.cluster_id].peaks])
        lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
        ys = (y >= lower[0] - 0.6) & (y <= upper[0] + 0.6)
        xs = (x >= lower[1] - 0.18) & (x <= upper[1] + 0.18)
        ax.contour(x[xs], y[ys], data[np.ix_(ys, xs)], levels=levels, colors="0.7", linewidths=0.7)
        draw_contours(ax, x[xs], y[ys], residual[np.ix_(ys, xs)], levels)
        for name in outcome.peak_names:
            nitrogen, proton = parameters[name + ".F2.cs"], parameters[name + ".F3.cs"]
            ax.plot(proton, nitrogen, "k+", ms=4)
            ax.annotate(name, (proton, nitrogen), fontsize=7)
        ax.set_title(", ".join(outcome.peak_names), fontsize=9)
    fig.savefig(output / "worst-residuals.png", dpi=140)
    plt.close(fig)


def case_config(template, name, shape, tilted, eta_min, output):
    """Keep baseline observation masks and starting bounds identical."""
    config = template.model_copy(deep=True)
    config.fitting.lineshape = shape
    config.fitting.tilted = tilted
    config.parameters.defaults.patterns["*.*.eta"] = ParameterConstraint(
        value=0.0,
        min=eta_min,
        max=1.0,
    )
    if name != "recommended":
        config.fitting.refine_iterations = 3
        config.clustering.contour_factor = 5.0
        config.parameters.defaults.patterns["*.*.tilt"].min = -30.0
        config.parameters.defaults.patterns["*.*.tilt"].max = 30.0
    config.output.directory = output / name
    return config


def main():
    """Fit all comparison cases and save canonical results and residual maps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path, help="Bruker 2rr file or processing folder")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spectrum = resolve_spectrum_path(args.spectrum)
    peaklist = spectrum / "peaklist.xml"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = load_config(Path(__file__).with_name("recommended.toml"))
    metrics = []
    reference_mask = None
    for name, shape, tilted, eta_min in CASES:
        config = case_config(template, name, shape, tilted, eta_min, output)
        config.output.directory.mkdir(parents=True, exist_ok=True)
        save_config(config, config.output.directory / "config.toml")
        logging.info("Fitting %s...", name)
        loaded = load_data(spectrum, peaklist, None, config)
        fitted = run_fit(loaded, config, config.output.directory, workers=1)
        if fitted.summary.n_unusable:
            raise RuntimeError(f"{name} has unusable clusters; inspect the fit before comparing.")
        write_fit_run_outputs(
            fitted,
            loaded.spectra,
            config,
            {"spectrum": spectrum / "2rr", "peaklist": peaklist},
        )
        assert fitted.simulation_snapshot is not None
        model = simulate_final_outcome(
            fitted.outcome, fitted.simulation_snapshot, loaded.spectra.data
        )
        if model is None:
            raise RuntimeError(f"{name} did not produce a usable model.")
        mask = np.zeros(loaded.spectra.data.shape[1:], dtype=bool)
        for cluster in loaded.clusters:
            mask[tuple(cluster.grid_indices)] = True
        if reference_mask is None:
            reference_mask = mask.copy()
        elif name != "recommended" and not np.array_equal(mask, reference_mask):
            raise RuntimeError("Baseline models must use exactly the same observation mask.")
        residual = loaded.spectra.data - model
        y = loaded.spectra.params[1].pts2ppm(np.arange(model.shape[1]))
        x = loaded.spectra.params[2].pts2ppm(np.arange(model.shape[2]))
        np.savez_compressed(
            config.output.directory / "model.npz",
            model=model[0],
            residual=residual[0],
            fit_mask=mask,
            x=x,
            y=y,
            noise=loaded.noise,
        )
        statistics = fitted.outcome.statistics
        row = {
            "model": name,
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "n_peaks": fitted.summary.n_peaks,
            "n_clusters": fitted.summary.n_clusters,
            "n_converged": fitted.summary.n_converged,
            "n_observations": statistics.n_observations,
            "n_fitted_parameters": statistics.n_fitted_parameters,
            "chi_squared": statistics.chi_squared,
            "reduced_chi_squared": statistics.reduced_chi_squared,
            "bic": statistics.bic,
            "reference_mask_rms": float(np.sqrt(np.mean(residual[0][reference_mask] ** 2))),
            "full_rms": float(np.sqrt(np.mean(residual**2))),
            "noise": loaded.noise,
        }
        metrics.append(row)
        if name == "recommended":
            plot_results(output, loaded, fitted, model)
        logging.info("  full-spectrum RMS %.1f", row["full_rms"])
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
