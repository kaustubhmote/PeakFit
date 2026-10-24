"""Generated scientific fixtures exercise the public interface, without external datasets."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from nmrglue.fileio import bruker, fileiobase, pipe, sparky
from typer.testing import CliRunner

from peakfit.api import (
    AxisProcessing,
    FitCancelledError,
    PeakFitConfig,
    PeakSeed,
    SeriesAxis,
    SpectrumInput,
    extract_spectrum,
    fit_file,
    fit_spectrum,
    model_availability,
    plan_extraction,
    prepare_spectrum,
)
from peakfit.auto_pick.decision import calculate_dof_scale_from_header
from peakfit.cli.app import app
from peakfit.engine.algorithms import evaluation
from peakfit.engine.algorithms.clustering import segment_data
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import DQFCOSYWindow
from peakfit.engine.domain.params_scalar import Parameters
from peakfit.fit import api as implementation
from peakfit.fit import pipeline


def synthetic(ndim=2, dtype=np.float64, ratios=None):
    shape = tuple(23 + 4 * axis for axis in range(ndim))
    udic = {"ndim": ndim}
    positions = {}
    data = np.ones(shape)
    for axis, size in enumerate(shape):
        obs = (60.0, 150.0, 600.0)[axis]
        sw = 240.0 + 40 * axis
        center = (120.0, 55.0, 8.0)[axis]
        udic[axis] = {
            "size": size,
            "sw": sw,
            "obs": obs,
            "car": center * obs,
            "label": "15N" if axis < 2 else "1H",
            "freq": True,
            "time": False,
            "complex": False,
            "encoding": "direct" if axis == ndim - 1 else "states",
        }
        first = center + sw / (2 * obs)
        grid = first - np.arange(size) * sw / (size * obs)
        positions[axis] = center + 0.17 * sw / (size * obs)
        gaussian = np.exp(-4 * np.log(2) * ((grid - positions[axis]) * obs / (25 + axis)) ** 2)
        view = [1] * ndim
        view[axis] = size
        data *= gaussian.reshape(view)
    data *= 100
    if ratios is not None:
        data = np.stack([data * ratio for ratio in ratios], axis=0)
        udic = {"ndim": ndim + 1, **{k + 1: v for k, v in udic.items() if isinstance(k, int)}}
        udic[0] = {
            "size": len(ratios),
            "label": "relaxation",
            "freq": False,
            "time": True,
            "complex": False,
            "sw": 1.0,
            "obs": 1.0,
            "car": 0.0,
        }
        positions = {key + 1: value for key, value in positions.items()}
    return data.astype(dtype), udic, positions


def config(shape="gaussian"):
    result = PeakFitConfig(noise_level=0.1)
    result.fitting.lineshape = shape
    return result


def region(udic, series_axis=None):
    result = {}
    for axis in range(udic["ndim"]):
        if axis == series_axis:
            continue
        uc = fileiobase.uc_from_udic(udic, axis)
        grid = uc.ppm_scale()
        result[axis] = (grid[3], grid[-4])
    return result


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_actual_fit_all_frequency_ranks(ndim, dtype, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data, udic, positions = synthetic(ndim, dtype)
    original = data.copy()
    metadata = deepcopy(udic)
    cfg = config()
    settings = cfg.model_dump()
    source = SpectrumInput.from_array(data, udic, direct_axis=ndim - 1)
    result = fit_spectrum(
        source, peaks=[PeakSeed("stable", positions)], region_ppm=region(udic), config=cfg
    )
    assert result.outcome.overall_converged
    assert len(result.peaks) == 1
    assert result.peaks[0].id == "stable"
    assert result.peaks[0].amplitudes == pytest.approx((100,), rel=2e-5)
    for axis, position in positions.items():
        assert result.peaks[0].positions_ppm[axis] == pytest.approx(position, abs=1e-6)
    simulated = result.simulate()
    residual = result.residual()
    assert simulated is not None
    assert residual is not None
    assert simulated.shape == (1, *(size - 6 for size in data.shape))
    assert np.max(np.abs(residual)) < 2e-5
    np.testing.assert_array_equal(data, original)
    assert udic == metadata
    assert cfg.model_dump() == settings
    assert list(tmp_path.iterdir()) == []
    saved = simulated.copy()
    data[...] = -500
    udic[0]["car"] = -1
    np.testing.assert_array_equal(result.simulate(), saved)


def test_series_true_3d_nonleading_series_and_direct_axes():
    ratios = (1.0, 0.5, 1.8, -0.3)
    data, udic, positions = synthetic(3, ratios=ratios)
    # external order is [original direct frequency, second frequency, series, first frequency]
    permutation = (3, 2, 0, 1)
    data = data.transpose(permutation)
    udic = {"ndim": 4, **{i: udic[j] for i, j in enumerate(permutation)}}
    positions = {i: positions[j] for i, j in enumerate(permutation) if j != 0}
    series = SeriesAxis((20, 5, 20, 10), "ms", "relaxation", ("a", "b", "replicate-a", "c"))
    source = SpectrumInput.from_array(data, udic, direct_axis=0, series_axis=2, series=series)
    prepared = prepare_spectrum(source, peaks=[PeakSeed("series", positions)], config=config())
    assert prepared.extraction.axis_permutation == (2, 1, 3, 0)
    result = fit_spectrum(source, peaks=[PeakSeed("series", positions)], config=config())
    assert result.peaks[0].amplitudes == pytest.approx(np.array(ratios) * 100, rel=1e-5)
    assert result.extraction.series == series
    assert len([p for p in result.peaks[0].parameters if p.unit == "ppm"]) == 3
    assert result.peaks[0].positions_ppm == pytest.approx(positions, abs=1e-7)
    # Shared covariance is available on outcome; no fabricated series error broadcasting.
    evaluation = result.outcome.clusters[0].analytical_evaluation
    assert evaluation is not None
    assert evaluation.amplitudes.shape == (1, 4)
    if evaluation.amplitude_standard_errors.shape != (1, 4):
        assert result.peaks[0].amplitude_standard_errors is None


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_crop_coordinate_invariants_against_independent_and_nmrglue(ndim):
    data, udic, _ = synthetic(ndim)
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    bounds = region(udic)
    # Reverse individual bounds and test exact sampled endpoints.
    bounds[0] = bounds[0][::-1]
    extracted = extract_spectrum(source, bounds)
    assert extracted.spectra.data.dtype == data.dtype
    for axis in extracted.extraction.axes:
        original = udic[axis.external_axis]
        meta = extracted.extraction.cropped_udic[axis.external_axis]
        n = meta["size"]
        independent = (
            original["car"] / original["obs"]
            + original["sw"] / (2 * original["obs"])
            - (3 + np.arange(n)) * original["sw"] / (original["size"] * original["obs"])
        )
        actual = extracted.spectra.params[axis.internal_axis].pts2ppm(np.arange(n, dtype=float))
        np.testing.assert_allclose(actual, independent, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            fileiobase.uc_from_udic(
                dict(extracted.extraction.cropped_udic), axis.external_axis
            ).ppm_scale(),
            independent,
            atol=1e-12,
            rtol=0,
        )
        assert axis.fitting_slice == slice(3, original["size"] - 3)
        assert extracted.spectra.params[axis.internal_axis].nucleus == original["label"]
        assert np.diff(actual) == pytest.approx(np.diff(independent), abs=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_readonly_noncontiguous_ownership(dtype):
    data, udic, positions = synthetic(2, dtype)
    large = np.zeros((data.shape[0] * 2, data.shape[1] * 2), dtype=dtype)
    large[::2, ::2] = data
    view = large[::2, ::2]
    view.flags.writeable = False
    source = SpectrumInput.from_array(view, udic, direct_axis=1)
    prepared = extract_spectrum(source)
    assert prepared.spectra.data.dtype == dtype
    assert not np.shares_memory(prepared.spectra.data, view)
    result = fit_spectrum(source, peaks=[PeakSeed("peak", positions)], config=config())
    large[:] = 200
    assert result.peaks[0].amplitudes == pytest.approx((100,), rel=1e-5)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("fmt", ["pipe", "ucsf"])
def test_array_file_equivalence_with_generated_native_files(ndim, fmt, tmp_path):
    data, udic, positions = synthetic(ndim, np.float32)
    filename = tmp_path / ("spectrum.ft" + str(ndim) if fmt == "pipe" else "spectrum.ucsf")
    if fmt == "pipe":
        headers = pipe.create_dic(udic)
        if ndim > 2:
            headers["FDPIPEFLAG"] = 1
        # NMRPipe's integer CENTER convention shifts odd-sized grids by half
        # a point. Declare ORIG independently to match this fixture's UDIC.
        for axis in range(ndim):
            meta = udic[axis]
            fn = f"FDF{int(headers['FDDIMORDER'][ndim - 1 - axis])}"
            headers[fn + "ORIG"] = meta["car"] - meta["sw"] / 2 + meta["sw"] / meta["size"]
        pipe.write(str(filename), headers, data, overwrite=True)
    else:
        headers = sparky.create_dic(udic)
        sparky.write(str(filename), headers, data, overwrite=True)
    source = SpectrumInput.from_array(data, udic, direct_axis=ndim - 1)
    bounds = region(udic)
    # Native headers store float32 references. Use half-sample margins here;
    # exact sampled endpoints are protected separately on the array path.
    for axis in bounds:
        step = udic[axis]["sw"] / (udic[axis]["size"] * udic[axis]["obs"])
        bounds[axis] = (bounds[axis][0] + step / 2, bounds[axis][1] - step / 2)
    seeds = [PeakSeed("native", positions)]
    a = fit_spectrum(source, peaks=seeds, region_ppm=bounds, config=config())
    b = fit_file(filename, direct_axis=ndim - 1, peaks=seeds, region_ppm=bounds, config=config())
    assert b.extraction.original_slices == a.extraction.original_slices
    assert b.peaks[0].positions_ppm == pytest.approx(a.peaks[0].positions_ppm, abs=1e-5)
    assert b.peaks[0].amplitudes == pytest.approx(a.peaks[0].amplitudes, rel=2e-5)
    sim_a, sim_b = a.simulate(), b.simulate()
    assert sim_a is not None
    assert sim_b is not None
    np.testing.assert_allclose(sim_b, sim_a, atol=1e-5)
    assert "bounded" in b.extraction.loading
    if fmt == "ucsf":
        auto = prepare_spectrum(
            SpectrumInput.from_file(filename, direct_axis=ndim - 1), peaks=seeds, noise=0.1
        )
        assert all(model.selected == "pvoigt" for model in auto.models)
        with pytest.raises(ValueError, match="unavailable"):
            fit_file(filename, direct_axis=ndim - 1, peaks=seeds, config=config("no_apod"))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda u: u.update(ndim=3), "ndim"),
        (lambda u: u[0].update(size=200), "size"),
        (lambda u: u[0].update(sw=0), "positive"),
        (lambda u: u[0].update(obs=float("nan")), "finite obs"),
        (lambda u: u[0].pop("car"), "finite car"),
        (lambda u: u[0].update(label="X"), "placeholder"),
        (lambda u: u[0].update(time=True, freq=False), "time-domain"),
        (lambda u: u[0].update(complex=True), "complex"),
    ],
)
def test_invalid_metadata(change, match):
    data, udic, _ = synthetic(2)
    change(udic)
    with pytest.raises(ValueError, match=match):
        SpectrumInput.from_array(data, udic, direct_axis=1)


@pytest.mark.parametrize("kind", ["complex", "object", "nan", "empty"])
def test_invalid_data(kind):
    data, udic, _ = synthetic(2)
    if kind == "complex":
        data = data.astype(complex)
    elif kind == "object":
        data = data.astype(object)
    elif kind == "nan":
        data[0, 0] = float("nan")
    else:
        data = data[:0]
    with pytest.raises(ValueError, match=r"numeric|nonfinite|nonempty"):
        SpectrumInput.from_array(data, udic, direct_axis=1)


@pytest.mark.parametrize(
    ("bounds", "match"),
    [
        ({0: (1, 2)}, "every"),
        ({0: (1, 2), 1: (1, 2), 2: (1, 2)}, "every"),
        ({0: (1, 2), 1: (float("nan"), 2)}, "outside|finite"),
        ({0: (200, 201), 1: (55, 55.1)}, "outside"),
        ({0: (120.000001, 120.000002), 1: (55, 55.1)}, "no sampled"),
    ],
)
def test_invalid_regions(bounds, match):
    data, udic, _ = synthetic(2)
    source = SpectrumInput.from_array(data, udic, direct_axis=1)
    with pytest.raises(ValueError, match=match):
        extract_spectrum(source, bounds)


def test_no_nearest_rounding_and_no_wrapping():
    data, udic, _ = synthetic(1)
    grid = fileiobase.uc_from_udic(udic, 0).ppm_scale()
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    width = abs(grid[1] - grid[0])
    extracted = extract_spectrum(source, {0: (grid[3] - width * 0.1, grid[9] + width * 0.1)})
    assert extracted.extraction.original_slices == (slice(4, 9),)
    with pytest.raises(ValueError, match="outside"):
        extract_spectrum(source, {0: (grid[0] + width, grid[-1])})


def test_models_and_processing_history_survive_crop():
    data, udic, positions = synthetic(1)
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    prepared = prepare_spectrum(source, peaks=[PeakSeed("p", positions)], noise=0.1)
    assert prepared.models[0].selected == "pvoigt"
    assert "acquisition_time" in prepared.models[0].missing_fields
    with pytest.raises(ValueError, match="Missing processing"):
        prepare_spectrum(source, peaks=[PeakSeed("p", positions)], config=config("sp1"))
    proc = AxisProcessing(
        provenance="synthetic acquisition",
        acquisition_time=0.1,
        td_size=24,
        ft_size=64,
        apod_code=0,
        phase0_deg=0,
        phase1_deg=0,
    )
    source = SpectrumInput.from_array(data, udic, direct_axis=0, processing={0: proc})
    crop = prepare_spectrum(
        source, peaks=[PeakSeed("p", positions)], region_ppm=region(udic), config=config("no_apod")
    )
    param = crop.data.spectra.spectral_params[0]
    assert param.aq_time == 0.1
    assert param.ft_size == 64
    assert param.td_size == 24
    assert param.sw < udic[0]["sw"]
    assert crop.extraction.processing[0].provenance == "synthetic acquisition"


def test_halo_neighbors_boundaries_and_exclusions():
    data, udic, _positions = synthetic(1)
    grid = fileiobase.uc_from_udic(udic, 0).ppm_scale()
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    seeds = [
        PeakSeed("boundary", {0: grid[7]}),
        PeakSeed("halo", {0: grid[6]}),
        PeakSeed("outside", {0: grid[1]}),
    ]
    prepared = prepare_spectrum(
        source, peaks=seeds, region_ppm={0: (grid[7], grid[15])}, halo_points=1, config=config()
    )
    assert [d.included for d in prepared.seeds] == [True, True, False]
    assert prepared.seeds[1].boundary_axes == (0,)
    assert not prepared.seeds[1].in_analysis_region
    assert "outside" in prepared.seeds[2].reason
    neighbor = PeakSeed("tail", {0: grid[5]}, widths_hz={0: 25})
    prepared = prepare_spectrum(
        source, peaks=[seeds[0], neighbor], region_ppm={0: (grid[7], grid[15])}, config=config()
    )
    assert prepared.seeds[1].included
    assert prepared.extraction.original_slices[0].start == 5
    clipped = extract_spectrum(source, halo_points=2)
    assert clipped.extraction.axes[0].halo_clipped
    with pytest.raises(ValueError, match="No usable"):
        prepare_spectrum(
            source, peaks=[seeds[2]], region_ppm={0: (grid[7], grid[15])}, config=config()
        )


def test_negative_peaks_noise_region_and_explicit_noise():
    data, udic, positions = synthetic(1)
    rng = np.random.default_rng(13)
    data = -data + rng.normal(0, 0.05, data.shape)
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    seed = PeakSeed("negative", positions, sign=-1)
    result = fit_spectrum(source, peaks=[seed], noise=0.05, config=config())
    assert result.peaks[0].amplitudes is not None
    assert result.peaks[0].amplitudes[0] < 0
    assert result.noise_source == "user-provided"
    grid = fileiobase.uc_from_udic(udic, 0).ppm_scale()
    no_config_noise = PeakFitConfig()
    no_config_noise.fitting.lineshape = "gaussian"
    prepared = prepare_spectrum(
        source,
        peaks=[seed],
        region_ppm={0: (grid[6], grid[16])},
        noise_region_ppm={0: (grid[0], grid[4])},
        config=no_config_noise,
    )
    assert "separate source" in prepared.data.noise_source


def test_no_file_export_final_authority_cancellation_and_resource_limits(tmp_path):
    data, udic, positions = synthetic(1)
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    seed = PeakSeed("export", positions)
    events = []
    result = fit_spectrum(source, peaks=[seed], config=config(), progress=events.append)
    assert events[0].selected_points == data.size
    files = result.export(tmp_path / "export")
    summary = json.loads(files["summary_fit"].read_text())
    assert summary["noise"] == result.outcome.noise
    assert not list(tmp_path.rglob("*.pkl"))
    destination = tmp_path / "cancelled"
    cancelled = False

    def stop(_event):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(FitCancelledError):
        fit_spectrum(
            source,
            peaks=[seed],
            config=config(),
            output_dir=destination,
            progress=stop,
            cancel=lambda: cancelled,
        )
    assert not destination.exists()
    with pytest.raises(MemoryError):
        plan_extraction(source, max_memory_bytes=1)
    with pytest.raises(ValueError, match="workers"):
        fit_spectrum(source, peaks=[seed], config=config(), workers=-1)


def test_series_validation_and_unsupported_models():
    data, udic, positions = synthetic(3, ratios=(1, 2, 3))
    with pytest.raises(ValueError, match="series_axis"):
        SpectrumInput.from_array(data, udic, direct_axis=3)
    series = SeriesAxis((1, 2, 3), "s", "time")
    source = SpectrumInput.from_array(data, udic, direct_axis=3, series_axis=0, series=series)
    bounds = region(udic, 0)
    bounds[0] = (1, 2)
    with pytest.raises(ValueError, match="series axis"):
        extract_spectrum(source, bounds)
    with pytest.raises(ValueError, match="exactly two"):
        fit_spectrum(source, peaks=[PeakSeed("p", positions)], config=config("dqfcosy"))
    cfg = config()
    cfg.fitting.tilted = True
    with pytest.raises(ValueError, match="exactly two"):
        fit_spectrum(source, peaks=[PeakSeed("p", positions)], config=cfg)
    with pytest.raises(ValueError, match="explicit compatible units"):
        SeriesAxis((1, 2, 3), "ppm", "time")
    with pytest.raises(ValueError, match="exactly"):
        SpectrumInput.from_array(
            data, udic, direct_axis=3, series_axis=0, series=SeriesAxis((1, 2), "s", "time")
        )


def test_experimental_auto_pick_is_explicit_and_headless():
    data, udic, _ = synthetic(2)
    source = SpectrumInput.from_array(data, udic, direct_axis=1)
    cfg = config()
    cfg.auto_peak.max_clusters = 2
    cfg.auto_peak.max_peaks_per_roi = 2
    with pytest.raises(ValueError, match="explicitly"):
        fit_spectrum(source, config=cfg)
    result = fit_spectrum(source, config=cfg, auto_pick=True)
    assert result.peaks
    assert all(peak.seed.provenance == "experimental auto picker" for peak in result.peaks)
    assert result.peaks[0].amplitudes == pytest.approx((100,), rel=1e-3)


def test_shared_widths_overlap_labels_and_seed_starts():
    data, udic, positions = synthetic(1)
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    second = {0: positions[0] + 0.4}
    seeds = [
        PeakSeed("left", positions, widths_hz={0: 22}, label="L", provenance="manual assignment"),
        PeakSeed("right", second, widths_hz={0: 26}, sign=-1),
    ]
    prepared = prepare_spectrum(source, peaks=seeds, config=config())
    params = Parameters.from_peaks(prepared.data.peaks, fixed=False)
    params["left.F2.lw"].value = 22
    params["right.F2.lw"].value = 26
    points = [np.arange(data.size)]
    values = Cluster.evaluate_peaks(prepared.data.peaks, params, points)
    signal = 80 * values[0] - 35 * values[1]
    fitted = fit_spectrum(
        SpectrumInput.from_array(signal, udic, direct_axis=0), peaks=seeds, config=config()
    )
    by_id = {peak.id: peak for peak in fitted.peaks}
    assert by_id["left"].amplitudes == pytest.approx((80,), rel=1e-5)
    assert by_id["right"].amplitudes == pytest.approx((-35,), rel=1e-5)
    assert by_id["left"].seed.label == "L"
    assert by_id["left"].seed.provenance == "manual assignment"


def test_crop_preserves_physics_not_a_new_alias_period_or_zero_filling():
    data, udic, _ = synthetic(2)
    processing = {
        axis: AxisProcessing(
            provenance="known acquisition",
            acquisition_time=0.05,
            td_size=4,
            ft_size=udic[axis]["size"],
            apod_code=0,
            phase0_deg=0,
            phase1_deg=0,
        )
        for axis in range(2)
    }
    source = SpectrumInput.from_array(data, udic, direct_axis=1, processing=processing)
    full, crop = extract_spectrum(source), extract_spectrum(source, region(udic))
    assert calculate_dof_scale_from_header(full.spectra) == calculate_dof_scale_from_header(
        crop.spectra
    )
    for index in range(len(crop.extraction.axes)):
        original = full.spectra.spectral_params[index]
        selected = crop.spectra.spectral_params[index]
        center = original.pts2ppm(1.0)
        indices = np.arange(selected.size, dtype=float)
        a = original.to_dimension_context("original").compute_offsets(
            indices + 3, np.array([center])
        )
        b = selected.to_dimension_context("crop").compute_offsets(indices, np.array([center]))
        np.testing.assert_allclose(a[0], b[0], atol=1e-10)
    edge_data = np.array([[10, 0, 0, 0, 10]], dtype=float)
    segmented = segment_data(edge_data, 5, [], periodic_axes=[])
    assert segmented[0] != segmented[-1]


def test_series_exclusions_keep_original_member_map():
    data, udic, positions = synthetic(1, ratios=(1, 2, 3, 4))
    series = SeriesAxis((20, 10, 20, 5), "ms", "relaxation", ("a", "b", "replicate-a", "c"))
    source = SpectrumInput.from_array(data, udic, direct_axis=1, series_axis=0, series=series)
    cfg = config()
    cfg.exclude_planes = [1]
    result = fit_spectrum(source, peaks=[PeakSeed("series", positions)], config=cfg)
    assert result.extraction.series_indices == (0, 2, 3)
    assert result.extraction.series is not None
    assert result.extraction.series.values == (20, 20, 5)
    assert result.extraction.series.ids == ("a", "replicate-a", "c")
    assert result.peaks[0].amplitudes == pytest.approx((100, 300, 400), rel=1e-5)
    assert cfg.exclude_planes == [1]


def test_processed_bruker_scaling_coordinates_and_capabilities(tmp_path):
    processed = tmp_path / "sample" / "pdata" / "1"
    processed.mkdir(parents=True)
    n = 32
    procs = {
        "SI": n,
        "FTSIZE": n,
        "XDIM": n,
        "SF": 600.0,
        "OFFSET": 8.2,
        "SW_p": 240.0,
        "WDW": 0,
        "FT_mod": 6,
        "NC_proc": -2,
        "BYTORDP": 0,
        "DTYPP": 0,
        "PHC0": 0,
        "PHC1": 0,
    }
    proc2s = {**procs, "SF": 60.0, "OFFSET": 122.0, "MC2": 4}
    acquisition = {"TD": 64, "SW_h": 240.0, "SFO1": 600.0, "AQ_mod": 3, "NUC1": "<1H>"}
    indirect = {"TD": 32, "SW": 4.0, "SW_h": 240.0, "SFO1": 60.0, "FnMODE": 5, "NUC1": "<15N>"}
    for path, parameters in [
        (processed / "procs", procs),
        (processed / "proc2s", proc2s),
        (tmp_path / "sample" / "acqus", acquisition),
        (tmp_path / "sample" / "acqu2s", indirect),
    ]:
        bruker.write_jcamp(
            {"_comments": [], "_coreheader": [], **parameters}, str(path), overwrite=True
        )
    y, x = np.indices((n, n))
    raw = np.rint(10000 * np.exp(-((y - 15.2) ** 2 + (x - 15.4) ** 2) / 9)).astype("<i4")
    (processed / "2rr").write_bytes(raw.tobytes())
    source = SpectrumInput.from_file(processed)
    extracted = extract_spectrum(source)
    np.testing.assert_array_equal(extracted.spectra.data[0], raw / 4)
    array = SpectrumInput.from_array(
        raw.astype(float) / 4, source.udic, direct_axis=1, processing=source.processing
    )
    crop_file, crop_array = (
        extract_spectrum(source, region(source.udic)),
        extract_spectrum(array, region(source.udic)),
    )
    np.testing.assert_array_equal(crop_file.spectra.data, crop_array.spectra.data)
    assert "full processed Bruker" in crop_file.extraction.loading
    assert crop_file.extraction.native_headers["procs"]["NC_proc"] == -2
    assert crop_file.spectra.spectral_params[1].aq_time == 32 / 240
    assert all(model.selected == "no_apod" for model in model_availability(source))


def test_processing_rejection_before_crop_allocation_and_preflight_budget(monkeypatch):
    data, udic, positions = synthetic(1)
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    report = model_availability(source)
    assert "no_apod" in report[0].unavailable

    def allocation_forbidden(*_args, **_kwargs):
        pytest.fail("must reject processing metadata before allocating a crop")

    monkeypatch.setattr(implementation, "extract_spectrum", allocation_forbidden)
    with pytest.raises(ValueError, match="Missing processing"):
        fit_spectrum(source, peaks=[PeakSeed("p", positions)], config=config("no_apod"))
    assert plan_extraction(source).resources.data_bytes == data.nbytes


def test_api_imports_without_gui_and_exports_include_map(tmp_path):
    code = """
import sys
class BlockGUI:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'PySide6'}:
            raise ImportError('GUI imported: ' + fullname)
sys.meta_path.insert(0, BlockGUI())
from peakfit.api import fit_spectrum, SpectrumInput
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    data, udic, positions = synthetic(1)
    cfg = config()
    cfg.output.save_simulated = True
    result = fit_spectrum(
        SpectrumInput.from_array(data, udic, direct_axis=0),
        peaks=[PeakSeed("p", positions)],
        config=cfg,
        region_ppm=region(udic),
        output_dir=tmp_path,
    )
    provenance = json.loads((tmp_path / "metadata/extraction.json").read_text())
    assert provenance["extraction"]["original_slices"][0] == {"start": 3, "stop": 20, "step": None}
    assert provenance["seeds"][0]["seed"]["id"] == "p"
    snapshot = np.load(tmp_path / "simulated_region.npz")
    simulated = result.simulate()
    assert simulated is not None
    np.testing.assert_array_equal(snapshot["simulated"], simulated)


def test_region_cli_reuses_public_pipeline_and_rejects_duplicates(tmp_path):
    data, udic, positions = synthetic(2, np.float32)
    filename = tmp_path / "spectrum.ucsf"
    sparky.write(str(filename), sparky.create_dic(udic), data, overwrite=True)
    peaklist = tmp_path / "peaks.list"
    peaklist.write_text("p " + " ".join(str(value) for value in positions.values()) + "\n")
    bounds = region(udic)
    args = [
        "fit",
        str(filename),
        str(peaklist),
        "--headless",
        "--lineshape",
        "gaussian",
        "--noise",
        "0.1",
        "--direct-axis",
        "1",
        "--output",
        str(tmp_path / "results"),
    ]
    for axis, pair in bounds.items():
        args.extend(["--region-ppm", f"{axis}:{pair[0]}:{pair[1]}"])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert list((tmp_path / "results").rglob("fit.json"))
    bad = CliRunner().invoke(app, [*args, "--region-ppm", "0:1:2"])
    assert bad.exit_code != 0
    assert "Duplicate region" in bad.output


def test_complete_explicit_dqf_processing_fits_without_native_headers():
    data, udic, positions = synthetic(2)
    window = DQFCOSYWindow(acquisition_time=0.1, points=24)
    processing = {
        axis: AxisProcessing(provenance="known DQF acquisition", dqfcosy_window=window)
        for axis in range(2)
    }
    source = SpectrumInput.from_array(
        np.zeros_like(data), udic, direct_axis=1, processing=processing
    )
    cfg = config("dqfcosy")
    cfg.fitting.fix_positions = True
    seeds = [PeakSeed("dqf", positions)]
    prepared = prepare_spectrum(source, peaks=seeds, config=cfg)
    params = Parameters.from_peaks(prepared.data.peaks, fixed=False)
    indices = [axis.ravel() for axis in np.indices(data.shape)]
    signal = 100 * Cluster.evaluate_peaks(prepared.data.peaks, params, indices)[0].reshape(
        data.shape
    )
    result = fit_spectrum(
        SpectrumInput.from_array(signal, udic, direct_axis=1, processing=processing),
        peaks=seeds,
        config=cfg,
    )
    assert result.peaks[0].amplitudes == pytest.approx((100,), rel=1e-5)
    assert cfg.fitting.dqfcosy_windows == {}


def test_native_pipe_capability_and_ext_history_policy(tmp_path):
    data, udic, _ = synthetic(2, np.float32)
    native = pipe.create_dic(udic)
    for fn in ("FDF1", "FDF2"):
        native[fn + "APOD"] = 16
        native[fn + "TDSIZE"] = 16
    filename = tmp_path / "complete.ft2"
    pipe.write(str(filename), native, data, overwrite=True)
    source = SpectrumInput.from_file(filename)
    report = model_availability(source)
    assert all("no_apod" in model.available for model in report)
    assert all("dqfcosy" in model.available for model in report)
    native["FDF2X1"] = 3
    filename = tmp_path / "extracted.ft2"
    pipe.write(str(filename), native, data, overwrite=True)
    cropped_source = SpectrumInput.from_file(filename)
    assert model_availability(cropped_source)[-1].selected == "pvoigt"
    assert "acquisition_time" in model_availability(cropped_source)[-1].missing_fields


def test_unusable_fit_preserves_classification_and_has_no_fake_scientific_values(monkeypatch):
    # Model an actual numerical optimizer failure at the final evaluation seam.
    original = evaluation.classify_optimizer_result

    def fail(*, cluster, result, noise):
        changed = replace(result, cost=float("nan"))
        return original(cluster=cluster, result=changed, noise=noise)

    # The pipeline imports this symbol directly.
    monkeypatch.setattr(pipeline, "classify_optimizer_result", fail)
    data, udic, positions = synthetic(1)
    result = fit_spectrum(
        SpectrumInput.from_array(data, udic, direct_axis=0),
        peaks=[PeakSeed("p", positions)],
        config=config(),
    )
    assert result.peaks[0].classification == "unusable"
    assert result.peaks[0].amplitudes is None
    assert result.peaks[0].positions_ppm == {}
    assert result.outcome.clusters[0].unusable_reason
    assert result.simulate() is None


def test_cancellation_after_optimizer_progress_never_exports(tmp_path):
    data, udic, positions = synthetic(1)
    source = SpectrumInput.from_array(data, udic, direct_axis=0)
    cancelled = False
    seen = []

    def progress(event):
        nonlocal cancelled
        seen.append(type(event).__name__)
        if type(event).__name__ == "FitResult":
            cancelled = True

    with pytest.raises(FitCancelledError):
        fit_spectrum(
            source,
            peaks=[PeakSeed("p", positions)],
            config=config(),
            progress=progress,
            cancel=lambda: cancelled,
            output_dir=tmp_path / "results",
        )
    assert "FitResult" in seen
    assert not (tmp_path / "results").exists()
