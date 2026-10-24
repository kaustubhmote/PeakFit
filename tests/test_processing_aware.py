"""Forward-model checks against independent FIDs and FFT processing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest
from nmrglue.fileio import pipe
from nmrglue.process import pipe_proc, proc_base

from peakfit.cli.commands.fit_setup import build_fit_config
from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters, get_shape_names
from peakfit.engine.domain.window import ProcessingWindow
from peakfit.engine.fitting.optimizers import fit_cluster
from peakfit.engine.lineshapes.create import create_shapes
from peakfit.io.readers.spectrum import (
    _bruker_processing_window,
    _pipe_processing_window,
    read_spectral_parameters,
)


def make_spectrum(windows, *, size=33, spacing=2.0):
    pseudo = SpectralParameters(
        size=1,
        sw=1,
        obs=1,
        car=0,
        aq_time=1,
        apocode=0,
        apodq1=0,
        apodq2=0,
        apodq3=0,
        p180=False,
        direct=False,
        ft=False,
    )
    axes = [
        SpectralParameters(
            size=size,
            sw=size * spacing,
            obs=100 * (axis + 1),
            car=spacing,
            aq_time=window.points * window.dwell,
            apocode=0,
            apodq1=0,
            apodq2=0,
            apodq3=0,
            p180=False,
            direct=axis == len(windows) - 1,
            ft=True,
            label=f"F{axis + 2}",
            processing_window=window,
        )
        for axis, window in enumerate(windows)
    ]
    return Spectra(
        dic={},
        data=np.zeros((1, *([size] * len(windows)))),
        z_values=np.array([0.0]),
        params=[pseudo, *axes],
    )


def make_model(spectrum, kind="gaussian", *, tilt=False, phase=False):
    positions = [0.0] * spectrum.n_spectral_dims
    shapes = create_shapes(
        spectrum,
        FitConfig(
            include_window_functions=True,
            fit_phase=spectrum.get_dimension_labels() if phase else [],
        ),
        "P",
        positions,
        [kind] * len(positions),
    )
    peak = Peak(name="P", positions=np.array(positions), shapes=shapes, tilted=tilt)
    peak.set_cluster_id(1)
    params = peak.create_params()
    indices = [a.ravel() for a in np.indices(spectrum.data.shape[1:])]
    cluster = Cluster(1, [peak], indices, np.zeros((indices[0].size, 1)))
    return peak, params, cluster


@pytest.mark.parametrize(
    ("code", "q"),
    [(0, (0, 0, 0)), (1, (0.2, 0.9, 2)), (2, (4, 0, 0)), (3, (10, 20, 0.3)), (7, (-8, 0.25, 0))],
)
def test_pipe_window_samples_match_nmrglue(code, q):
    n, sw, first = 64, 1200.0, 0.7
    dic = {
        "FDF2SW": sw,
        "FDF2APOD": n,
        "FDF2APODCODE": code,
        "FDF2APODQ1": q[0],
        "FDF2APODQ2": q[1],
        "FDF2APODQ3": q[2],
        "FDF2C1": first - 1,
    }
    window = _pipe_processing_window(dic, "FDF2", 256)
    data = np.ones(n)
    if code == 1:
        expected = proc_base.sp(data, off=q[0], end=q[1], pow=q[2])
    elif code == 2:
        expected = proc_base.em(data, lb=q[0] / sw)
    elif code == 3:
        expected = proc_base.gm(data, g1=q[0] / sw, g2=q[1] / sw, g3=q[2])
    elif code == 7:
        a = np.pi * q[0] / sw
        expected = proc_base.gmb(data, a=a, b=-a / (2 * q[1] * n))
    else:
        expected = data
    expected[0] *= first
    np.testing.assert_allclose(window.samples(), expected, atol=1e-13)


@pytest.mark.parametrize("direct", [True, False])
def test_bruker_cropped_gaussian_window_uses_acquisition_sampling_and_ignores_lp(direct):
    acquisition = {"TD": 256, "SW_h": 4000.0, "AQ_mod": 3, "FnMODE": 5}
    processing = {
        "WDW": 2,
        "LB": -12,
        "GB": 0.3,
        "FCOR": 0.5,
        "TDeff": 1024,
        "SI": 4096,
        "SW_p": 250,
        "ME_mod": 2,
        "LPBIN": 1024,
    }
    window = _bruker_processing_window(acquisition, processing, direct=direct)
    assert window.points == 128
    assert window.dwell == 1 / 4000
    assert window.duration == 128 / 4000
    t = np.arange(128) / 4000
    expected = np.exp(12 * np.pi * t - 12 * np.pi * t**2 / (2 * 0.3 * 0.032))
    expected[0] *= 0.5
    np.testing.assert_allclose(window.samples(), expected)


def test_bruker_retained_sine_window_keeps_original_duration():
    window = _bruker_processing_window(
        {"TD": 256, "SW_h": 4000, "FnMODE": 5},
        {"WDW": 4, "SSB": 2, "TDeff": 128, "SI": 4096, "FCOR": 1},
        direct=False,
    )
    expected = np.cos(np.pi / 2 * np.arange(64) / 128) ** 2
    np.testing.assert_allclose(window.samples(), expected)


@pytest.mark.parametrize("kind", ["none", "em", "gmb", "sine"])
@pytest.mark.parametrize("first", [0.5, 1])
@pytest.mark.parametrize("doublet", [False, True])
def test_lorentzian_matches_independent_one_sided_complex_fid(kind, first, doublet):
    window = ProcessingWindow(
        kind=kind,
        points=80,
        dwell=0.001,
        duration=0.08,
        q1=-4 if kind == "gmb" else 0.2 if kind == "sine" else 3,
        q2=0.3 if kind == "gmb" else 0.95,
        q3=2,
        first_point=first,
    )
    spectrum = make_spectrum([window])
    _, params, cluster = make_model(
        spectrum, "lorentzian_doublet" if doublet else "lorentzian", phase=True
    )
    params["P.F2.lw"].value = 17.5
    params["P.F2.cs"].value = 0.031
    params["cluster_1.F2.phase"].value = 13
    t = np.arange(80) * 0.001
    frequency = spectrum.params[1].pts2hz(np.arange(33)) - 3.1
    fid = np.exp(-np.pi * 17.5 * t) * window.samples()
    if doublet:
        coupling = params["P.F2.j"].value
        fid *= 2 * np.cos(np.pi * coupling * t)
    expected = (
        0.001
        * np.pi
        * 17.5
        * (np.exp(-2j * np.pi * frequency[:, None] * t) @ fid * np.exp(13j * np.pi / 180)).real
    )
    np.testing.assert_allclose(cluster.evaluate(params)[0], expected, atol=3e-14)


@pytest.mark.parametrize(
    "kind",
    [
        "gaussian",
        "lorentzian",
        "pvoigt",
        "gaussian_doublet",
        "lorentzian_doublet",
        "pvoigt_doublet",
    ],
)
@pytest.mark.parametrize("tilt", [False, True])
def test_processed_analytic_jacobian_matches_finite_differences(kind, tilt):
    windows = [
        ProcessingWindow(kind="gmb", points=24, dwell=0.002, duration=0.048, q1=-6, q2=0.4),
        ProcessingWindow(kind="sine", points=30, dwell=0.001, duration=0.03, q1=0.3, q2=1, q3=2),
    ]
    spectrum = make_spectrum(windows, size=9, spacing=3)
    _, params, cluster = make_model(spectrum, kind, tilt=tilt, phase=True)
    if tilt:
        params["P.F2-F3.tilt"].value = 23.4
    for axis in ("F2", "F3"):
        params[f"P.{axis}.cs"].value = 0.017
        params[f"cluster_1.{axis}.phase"].value = 7
        if kind.startswith("pvoigt"):
            params[f"P.{axis}.eta"].value = -0.2
    names = params.get_vary_names()
    jac = np.zeros((len(names), cluster.n_points))
    cluster.evaluate(params, d_matrix=jac, param_map=ParameterMap.from_names(names))
    for i, name in enumerate(names):
        old = params[name].value
        step = 1e-6 if params[name].unit == "ppm" else 1e-4
        params[name].value = old + step
        plus = cluster.evaluate(params)[0]
        params[name].value = old - step
        minus = cluster.evaluate(params)[0]
        params[name].value = old
        np.testing.assert_allclose(
            jac[i], (plus - minus) / (2 * step), rtol=2e-5, atol=2e-8, err_msg=name
        )


def test_tilted_gaussian_matches_independent_frequency_domain_fft_then_window():
    n, sw, width_x, width_y, angle = 256, 512, 18.0, 40.0, 0.41
    window_x = ProcessingWindow(kind="em", points=64, dwell=1 / sw, duration=64 / sw, q1=7)
    window_y = ProcessingWindow(
        kind="gmb", points=64, dwell=1 / sw, duration=64 / sw, q1=-8, q2=0.35
    )
    frequencies = np.fft.fftfreq(n, d=1 / sw)
    fx, fy = np.meshgrid(frequencies, frequencies, indexing="ij")
    # Both axes use 100 MHz for this independent reference.
    u = fx * np.cos(angle) + fy * np.sin(angle)
    v = -fx * np.sin(angle) + fy * np.cos(angle)
    intrinsic = np.exp(-4 * np.log(2) * ((u / width_x) ** 2 + (v / width_y) ** 2))
    coefficients = np.fft.ifft2(intrinsic)
    windows = []
    for win in (window_x, window_y):
        w = np.zeros(n)
        w[:64] = win.samples()
        w[0] *= 2
        w[-63:] = w[1:64][::-1]
        windows.append(w)
    expected = np.fft.fft2(coefficients * windows[0][:, None] * windows[1][None, :]).real
    spectrum = make_spectrum([window_x, window_y], size=n, spacing=sw / n)
    for axis in (1, 2):
        payload = spectrum.params[axis].model_dump()
        payload.update(obs=100, car=0)
        spectrum.params[axis] = SpectralParameters.model_validate(payload)
    _, params, cluster = make_model(spectrum, tilt=True)
    params["P.F2.lw"].value = width_x
    params["P.F3.lw"].value = width_y
    params["P.F2-F3.tilt"].value = np.rad2deg(angle)
    actual = cluster.evaluate(params)[0].reshape(n, n)
    # make_spectrum's descending coordinate grid includes +SW/2, ending at -SW/2+df.
    order = (-np.arange(n) + n // 2) % n
    np.testing.assert_allclose(actual, expected[np.ix_(order, order)], atol=5e-13)


def test_cropped_pipe_forward_model_matches_full_processed_spectrum():
    dic = pipe.create_empty_dic()
    dic.update(
        {
            "FDDIMORDER": [2, 1, 3, 4],
            "FDDIMCOUNT": 1,
            "FDSIZE": 256,
            "FDF2SW": 1024,
            "FDF2OBS": 100,
            "FDF2ORIG": -508,
            "FDF2FTFLAG": 1,
            "FDF2FTSIZE": 256,
            "FDF2APOD": 64,
            "FDF2APODCODE": 3,
            "FDF2APODQ1": 8,
            "FDF2APODQ2": 20,
            "FDF2APODQ3": 0.2,
            "FDF2C1": -0.5,
            "FDF2CENTER": 129,
            "FDF2CAR": 0,
            "FDF2QUADFLAG": 1,
        }
    )
    full_params = read_spectral_parameters(dic, np.zeros((1, 256)))
    full = Spectra(dic=dic, data=np.zeros((1, 256)), params=full_params, z_values=np.array([0.0]))
    _, params, cluster = make_model(full, "lorentzian")
    model = cluster.evaluate(params)[0]
    cropped_dic, cropped_data = pipe_proc.ext(dic.copy(), model, x1=97, xn=160, sw=True)
    cropped_params = read_spectral_parameters(cropped_dic, cropped_data[None, :])
    crop = Spectra(
        dic=cropped_dic, data=cropped_data[None, :], params=cropped_params, z_values=np.array([0.0])
    )
    _, crop_params, cropped_cluster = make_model(crop, "lorentzian")
    assert cropped_params[1].processing_window == full_params[1].processing_window
    assert cropped_params[1].sw == 256
    np.testing.assert_allclose(cropped_cluster.evaluate(crop_params)[0], model[96:160], atol=1e-13)


def test_optimizer_recovers_intrinsic_width_and_tilt_after_processing():
    windows = [
        ProcessingWindow(kind="em", points=64, dwell=0.002, duration=0.128, q1=12),
        ProcessingWindow(kind="gmb", points=64, dwell=0.001, duration=0.064, q1=-8, q2=0.4),
    ]
    spectrum = make_spectrum(windows, size=25, spacing=3)
    _, truth, empty = make_model(spectrum, tilt=True)
    truth["P.F2.lw"].value = 18
    truth["P.F3.lw"].value = 32
    truth["P.F2-F3.tilt"].value = 27
    cluster = Cluster(1, empty.peaks, empty.grid_indices, empty.evaluate(truth).T * 100)
    initial = truth.copy(deep=True)
    for parameter in initial.values():
        parameter.vary = parameter.unit != "ppm"
    initial["P.F2.lw"].value = 23
    initial["P.F3.lw"].value = 25
    initial["P.F2-F3.tilt"].value = 12
    result = fit_cluster(initial, cluster, noise=0.1, max_nfev=100)
    assert result.success
    for name in ("P.F2.lw", "P.F3.lw", "P.F2-F3.tilt"):
        assert result.params[name].value == pytest.approx(truth[name].value, abs=0.002)


def test_include_flag_overrides_toml_and_auto_uses_intrinsic_pvoigt(tmp_path: Path):
    config = tmp_path / "fit.toml"
    config.write_text('[fitting]\nlineshape = "gaussian"\n')
    cfg = build_fit_config(
        config,
        None,
        None,
        True,
        None,
        None,
        "auto",
        False,
        1,
        False,
        False,
        False,
        False,
        None,
        "varpro",
        include_window_functions=True,
    )
    assert cfg.fitting.include_window_functions
    assert cfg.fitting.lineshape == "gaussian"
    spectrum = make_spectrum([ProcessingWindow(kind="none", points=20, dwell=0.001, duration=0.02)])
    assert get_shape_names(FitConfig(include_window_functions=True), spectrum) == ["pvoigt"]
    assert get_shape_names(FitConfig(), spectrum) == ["no_apod"]


def test_missing_metadata_and_already_apodized_models_fail_clearly():
    spectrum = make_spectrum([ProcessingWindow(kind="none", points=20, dwell=0.001, duration=0.02)])
    with pytest.raises(ValueError, match="intrinsic shapes"):
        make_model(spectrum, "sp1")
    spectrum.params[1].processing_window = None
    spectrum.params[1].processing_window_error = "unsupported Bruker WDW=5"
    with pytest.raises(ValueError, match="WDW=5"):
        make_model(spectrum)
