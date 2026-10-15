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


def decode_contours(payload: bytes):
    count = struct.unpack_from("<I", payload)[0]
    offset = 4
    result = {}
    for _ in range(count):
        height, positive_count, negative_count = struct.unpack_from("<dII", payload, offset)
        offset += 16
        sections = []
        for point_count in (positive_count, negative_count):
            sections.append(
                np.frombuffer(payload, dtype="<f4", count=point_count * 2, offset=offset).reshape(
                    -1, 2
                )
            )
            offset += point_count * 8
        result[height] = sections
    assert offset == len(payload)
    return result


def test_binary_contours_keep_signs_and_sample_coordinates(viewer: SpectrumViewer) -> None:
    levels = [float(5 * 1.4**index) for index in range(25)]
    payload = viewer.contours(0, "experimental", levels)
    decoded = decode_contours(payload)
    assert list(decoded) == levels
    positive = np.concatenate([sections[0] for sections in decoded.values()])
    negative = np.concatenate([sections[1] for sections in decoded.values()])
    assert np.nanmin(positive[:, 0]) == pytest.approx(5.0)
    assert np.nanmax(negative[:, 0]) == pytest.approx(3.0)
    assert np.nanmin(positive[:, 1]) == 0
    assert np.nanmax(positive[:, 1]) == 6
    assert viewer.contours(0, "experimental", levels) == payload


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


@pytest.mark.parametrize("level", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_contour_levels(viewer: SpectrumViewer, level: float) -> None:
    with pytest.raises(ValueError, match="positive, finite contour heights"):
        viewer.contours(0, "experimental", [level])


@pytest.mark.parametrize(
    ("index", "layer"), [(-1, "experimental"), (2, "experimental"), (0, "residual"), (0, "unknown")]
)
def test_rejects_unavailable_planes_and_layers(viewer: SpectrumViewer, index, layer) -> None:
    with pytest.raises(ValueError, match=r"Plane index|Requested layer"):
        viewer.plane(index, layer)


def test_contour_cache_is_bounded(viewer: SpectrumViewer, monkeypatch: pytest.MonkeyPatch) -> None:
    viewer.contours(0, "experimental", [5])
    size = viewer._cache_size
    monkeypatch.setattr(spectra_viewer, "_CACHE_BYTES", size + 10)
    viewer.contours(1, "experimental", [5])
    assert viewer._cache_size == size
    assert (0, "experimental", 5) not in viewer._cache


def test_constant_spectrum_has_valid_empty_contours() -> None:
    viewer = SpectrumViewer(NMRData("zero", {}, np.zeros((1, 3, 4)), (9, 6), (130, 100)))
    assert viewer.noise > 0
    decoded = decode_contours(viewer.contours(0, "experimental", [5]))
    assert decoded[5][0].shape == (0, 2)
    assert decoded[5][1].shape == (0, 2)


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
            base + prefix + "contours?plane=0&layer=experimental&levels=5,7", timeout=5
        ) as response:
            assert response.headers["Content-Type"] == "application/octet-stream"
            assert response.read() == viewer.contours(0, "experimental", [5, 7])
        for route, code in [
            ("/metadata", 404),
            (prefix + "../pyproject.toml", 404),
            (prefix + "contours?plane=0&layer=residual&levels=5", 400),
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


@pytest.mark.parametrize("levels", [[], [1] * 26, [5, 5]])
def test_rejects_invalid_contour_batches(viewer: SpectrumViewer, levels) -> None:
    with pytest.raises(ValueError, match=r"Request between|must be unique"):
        viewer.contours(0, "experimental", levels)


def test_overlapping_levels_only_extract_the_added_height(
    viewer: SpectrumViewer, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = spectra_viewer.contourpy.contour_generator
    calculated = []

    class RecordingGenerator:
        def __init__(self, **kwargs):
            self.generator = original(**kwargs)

        def lines(self, height):
            calculated.append(height)
            return self.generator.lines(height)

    monkeypatch.setattr(spectra_viewer.contourpy, "contour_generator", RecordingGenerator)
    levels = [float(f"{5 * 1.4**index:.12g}") for index in range(26)]
    before = decode_contours(viewer.contours(0, "experimental", levels[:25]))
    assert len(calculated) == 50
    after = decode_contours(viewer.contours(0, "experimental", levels[1:]))
    assert calculated[50:] == [levels[-1], -levels[-1]]
    for height in levels[1:25]:
        np.testing.assert_array_equal(before[height][0], after[height][0])
        np.testing.assert_array_equal(before[height][1], after[height][1])
    viewer.contours(0, "experimental", levels[:25])
    assert len(calculated) == 52
