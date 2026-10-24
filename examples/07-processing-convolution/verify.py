"""Verify FFT convolution normalization and its analytic parameter derivatives."""

import sys

import numpy as np
from benchmark import ConvolutionCluster
from scipy.fft import fft

from peakfit.engine.domain.cluster import Cluster
from peakfit.engine.domain.config import FitConfig
from peakfit.engine.domain.param_map import ParameterMap
from peakfit.engine.domain.peaks import Peak
from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
from peakfit.engine.domain.window import ProcessingWindow
from peakfit.engine.lineshapes.create import create_shapes

window = ProcessingWindow(
    kind="gmb", points=20, dwell=0.002, duration=0.04, q1=-8, q2=0.3, first_point=0.5
)
base = {
    "size": 64,
    "sw": 500.0,
    "obs": 100.0,
    "car": 0.0,
    "aq_time": 0.04,
    "apocode": 0.0,
    "apodq1": 0.0,
    "apodq2": 0.0,
    "apodq3": 0.0,
    "p180": False,
    "direct": False,
    "ft": True,
}
params = [
    SpectralParameters.model_validate(base),
    SpectralParameters.model_validate({**base, "label": "F2", "processing_window": window}),
    SpectralParameters.model_validate(
        {**base, "obs": 200.0, "direct": True, "label": "F3", "processing_window": window}
    ),
]
spectrum = Spectra(dic={}, data=np.zeros((1, 64, 64)), z_values=np.array([0.0]), params=params)
positions = np.array([0.0, 0.0])
peak = Peak(
    name="P",
    positions=positions,
    shapes=create_shapes(
        spectrum,
        FitConfig(include_window_functions=True),
        "P",
        positions.tolist(),
        ["pvoigt", "pvoigt"],
    ),
    tilted=True,
)
peak.set_cluster_id(1)
parameters = peak.create_params()
parameters["P.F2-F3.tilt"].value = 15
indices = [a.ravel() for a in np.indices((64, 64))]
cluster = ConvolutionCluster(1, [peak], indices, np.zeros((4096, 1)))
cluster.prepare(2)
names = parameters.get_vary_names()
jac = np.zeros((len(names), 4096))
cluster.evaluate(parameters, d_matrix=jac, param_map=ParameterMap.from_names(names))
for i, name in enumerate(names):
    old = parameters[name].value
    step = 1e-6 if parameters[name].unit == "ppm" else 1e-4
    parameters[name].value = old + step
    plus = cluster.evaluate(parameters)[0]
    parameters[name].value = old - step
    minus = cluster.evaluate(parameters)[0]
    parameters[name].value = old
    np.testing.assert_allclose(jac[i], (plus - minus) / (2 * step), rtol=2e-5, atol=1e-8)
# Check the FFT normalization against an explicit sum of shifted intrinsic arrays.
rng = np.random.default_rng(1)
intrinsic = rng.normal(size=(128, 128))
kx = fft(cluster.filters[0])
ky = fft(cluster.filters[1])
expected = np.zeros_like(intrinsic, dtype=complex)
# One axis at a time avoids 128**2 Python rolls.
for shift, weight in enumerate(kx):
    expected += np.roll(intrinsic, shift, axis=0) * weight / 128
second = np.zeros_like(expected)
for shift, weight in enumerate(ky):
    second += np.roll(expected, shift, axis=1) * weight / 128
actual = cluster._convolve(intrinsic, indices)
ix = [
    np.round((p.pts2hz(g.astype(float)) - origin) / step).astype(int) % 128
    for p, g, origin, step in zip(
        spectrum.spectral_params, indices, cluster.origins, cluster.steps, strict=True
    )
]
np.testing.assert_allclose(actual, second.real[ix[0], ix[1]], atol=1e-13)
sys.stdout.write("Convolution normalization and all seven analytic derivatives verified.\n")

# Pure Gaussian input has negligible tails and sampling aliases on this lattice.
parameters["P.F2.eta"].value = 0.0
parameters["P.F3.eta"].value = 0.0
reference = Cluster(1, [peak], indices, np.zeros((4096, 1))).evaluate(parameters)
np.testing.assert_allclose(cluster.evaluate(parameters), reference, atol=1e-13)
# Selecting a small, noncentral spectral region retains the full acquisition period.
crop_indices = [a.ravel() for a in np.indices((32, 32))]
cropped_params = [params[0]]
for spec in spectrum.spectral_params:
    fields = spec.model_dump()
    fields.update(size=32, sw=250, car=spec.pts2hz(8.0) - 125)
    cropped_params.append(SpectralParameters.model_validate(fields))
cropped_spectrum = Spectra(
    dic={}, data=np.zeros((1, 32, 32)), params=cropped_params, z_values=np.array([0.0])
)
crop_peak = Peak(
    name="P",
    positions=positions,
    shapes=create_shapes(
        cropped_spectrum,
        FitConfig(include_window_functions=True),
        "P",
        positions.tolist(),
        ["pvoigt", "pvoigt"],
    ),
    tilted=True,
)
crop_peak.set_cluster_id(1)
crop = ConvolutionCluster(1, [crop_peak], crop_indices, np.zeros((1024, 1)))
crop.prepare(2)
np.testing.assert_allclose(
    crop.evaluate(parameters), reference.reshape(64, 64)[8:40, 8:40].reshape(1, -1), atol=1e-13
)
sys.stdout.write("Gaussian forward-model and noncentral crop equivalence verified.\n")
