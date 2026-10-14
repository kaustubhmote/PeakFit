"""Protect signed contours, residual arithmetic, HTTP delivery, and headless imports."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from threading import Thread
from typing import TYPE_CHECKING
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from peakfit.plot import spectra_viewer
from peakfit.plot.reconstruction import SpectraReconstructor
from peakfit.plot.spectra_viewer import NMRData, SpectrumViewer, create_viewer_server

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def viewer(monkeypatch: pytest.MonkeyPatch) -> SpectrumViewer:
    monkeypatch.setattr(spectra_viewer, "estimate_noise", lambda _data: 1.0)
    data = np.tile(np.linspace(-20, 20, 9), (2, 7, 1))
    return SpectrumViewer(NMRData("test.ft3", {}, data, (10, 2), (130, 100)))


def test_binary_contours_keep_signs_and_sample_coordinates(viewer: SpectrumViewer) -> None:
    payload = viewer.contours(0, "experimental", 5)
    positive_count = struct.unpack("<I", payload[:4])[0]
    points = np.frombuffer(payload, dtype="<f4", offset=4).reshape(-1, 2)
    positive = points[:positive_count]
    negative = points[positive_count:]
    assert np.nanmin(positive[:, 0]) == pytest.approx(5.0)
    assert np.nanmax(negative[:, 0]) == pytest.approx(3.0)
    assert np.nanmin(points[:, 1]) == 0
    assert np.nanmax(points[:, 1]) == 6
    assert np.isnan(positive).any()
    assert np.isnan(negative).any()
    assert viewer.contours(0, "experimental", 5) is payload


def test_simulation_and_residual_share_one_reconstruction(viewer: SpectrumViewer) -> None:
    calls = []

    class Reconstructor:
        def reconstruct_plane(self, index, shape):
            calls.append(index)
            return np.full(shape, 3.0)

    viewer.reconstructor = Reconstructor()
    simulated = viewer.plane(0, "simulated")
    residual = viewer.plane(0, "residual")
    np.testing.assert_array_equal(residual, viewer.data.data[0] - simulated)
    assert calls == [0]
    viewer.plane(1, "residual")
    assert calls == [0, 1]


@pytest.mark.parametrize("level", [0, -1, float("nan"), float("inf"), 1001])
def test_rejects_invalid_contour_levels(viewer: SpectrumViewer, level: float) -> None:
    with pytest.raises(ValueError, match="Contour level"):
        viewer.contours(0, "experimental", level)


@pytest.mark.parametrize(
    ("index", "layer"), [(-1, "experimental"), (2, "experimental"), (0, "residual"), (0, "unknown")]
)
def test_rejects_unavailable_planes_and_layers(viewer: SpectrumViewer, index, layer) -> None:
    with pytest.raises(ValueError, match=r"Plane index|Requested layer"):
        viewer.plane(index, layer)


def test_contour_cache_is_bounded(viewer: SpectrumViewer, monkeypatch: pytest.MonkeyPatch) -> None:
    first = viewer.contours(0, "experimental", 5)
    monkeypatch.setattr(spectra_viewer, "_CACHE_BYTES", len(first) + 10)
    second = viewer.contours(1, "experimental", 5)
    assert viewer._cache_size == len(second)
    assert (0, "experimental", 5, 1.4) not in viewer._cache


def test_constant_spectrum_has_valid_empty_contours() -> None:
    viewer = SpectrumViewer(NMRData("zero", {}, np.zeros((1, 3, 4)), (9, 6), (130, 100)))
    assert viewer.noise > 0
    assert viewer.contours(0, "experimental", 5) == struct.pack("<I", 0)


def test_indirect_peak_aliasing_uses_full_sweep_width(viewer: SpectrumViewer) -> None:
    width = (130 - 100) * 7 / 6
    np.testing.assert_allclose(
        viewer.data.unalias_y(np.array([120, 120 + width, 120 - width])), 120
    )


def test_http_serves_only_session_assets_and_valid_layers(viewer: SpectrumViewer) -> None:
    server, prefix = create_viewer_server(viewer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + prefix + "metadata", timeout=5) as response:
            metadata = json.load(response)
        assert metadata["shape"] == [2, 7, 9]
        assert metadata["layers"] == ["experimental"]
        for asset in ("", "viewer.js"):
            with urlopen(base + prefix + asset, timeout=5) as response:
                assert response.status == 200
                assert response.read()
        with urlopen(
            base + prefix + "contours?plane=0&layer=experimental&level=5", timeout=5
        ) as response:
            assert response.headers["Content-Type"] == "application/octet-stream"
            assert response.read() == viewer.contours(0, "experimental", 5)
        for route, code in [
            ("/metadata", 404),
            (prefix + "../pyproject.toml", 404),
            (prefix + "contours?plane=0&layer=residual&level=5", 400),
            (prefix + "contours?plane=0", 400),
        ]:
            with pytest.raises(HTTPError) as caught:
                urlopen(base + route, timeout=5)
            assert caught.value.code == code
            caught.value.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_and_browser_viewer_import_without_gui_libraries() -> None:
    code = """
import sys
class BlockGUI:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'PySide6'}:
            raise ImportError('GUI dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockGUI())
from peakfit.cli.app import app
from peakfit.plot.spectra_viewer import SpectrumViewer
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_missing_simulation_state_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "fit.json").write_text("{}")
    with pytest.raises(ValueError, match=r"metadata/fitting_state.pkl"):
        _ = SpectraReconstructor(tmp_path).state


@pytest.mark.parametrize("factor", [1, 0, float("nan"), float("inf"), 3.1])
def test_rejects_invalid_contour_multipliers(viewer: SpectrumViewer, factor: float) -> None:
    with pytest.raises(ValueError, match="Contour multiplier"):
        viewer.contours(0, "experimental", 5, factor)


def test_contour_multiplier_changes_spacing_and_cache(viewer: SpectrumViewer) -> None:
    payload = viewer.contours(0, "experimental", 5, 2)
    count = struct.unpack("<I", payload[:4])[0]
    points = np.frombuffer(payload, dtype="<f4", offset=4).reshape(-1, 2)[:count]
    xs = np.unique(points[np.isfinite(points[:, 0]), 0])
    np.testing.assert_allclose(xs, [5, 6])
    assert payload != viewer.contours(0, "experimental", 5, 1.4)
