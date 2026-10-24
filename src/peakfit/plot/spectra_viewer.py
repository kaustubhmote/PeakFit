"""Local browser spectrum viewer, with cached full-resolution contour geometry."""

from __future__ import annotations

import json
import secrets
import struct
import webbrowser
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Self, cast
from urllib.parse import parse_qs, urlsplit

import contourpy
import numpy as np

from peakfit.engine.algorithms.noise import estimate_noise
from peakfit.io.readers.spectrum import read_spectra

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd

    from peakfit.engine.domain.config import PlotConfig
    from peakfit.engine.domain.spectrum import Spectra, SpectralParameters
    from peakfit.plot.reconstruction import SpectraReconstructor
    from peakfit.shared.typing import FloatArray

_VIEWER_NDIM = 3
_MIN_GRID_SIZE = 2
_CONTOUR_COUNT = 25
_CACHE_BYTES = 64 * 1024 * 1024
_NOISE_SAMPLE_SIZE = 200_000


@dataclass
class NMRData:
    """Loaded pseudo-3D data and coordinates at the first and last sample."""

    filename: str
    dic: dict[str, Any]
    data: FloatArray
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    x_label: str = "F3"
    y_label: str = "F2"

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """Load an NMRPipe file or processed Bruker spectrum."""
        return cls.from_spectra(filename, read_spectra(Path(filename)))

    @classmethod
    def from_spectra(cls, filename: str, spectra: Spectra) -> Self:
        """Keep a normalized region's exact coordinate grid and axis identities."""
        data, xlim, ylim = cls._process_data(spectra.data, spectra.spectral_params)
        y_axis, x_axis = spectra.spectral_params
        return cls(
            filename,
            spectra.dic,
            data,
            xlim,
            ylim,
            x_axis.nucleus or x_axis.label,
            y_axis.nucleus or y_axis.label,
        )

    @staticmethod
    def _process_data(
        data: FloatArray, spectral_params: list[SpectralParameters]
    ) -> tuple[FloatArray, tuple[float, float], tuple[float, float]]:
        if data.ndim != _VIEWER_NDIM or len(spectral_params) != _VIEWER_NDIM - 1:
            raise ValueError("Spectrum viewer requires two spectral dimensions and a plane axis.")
        if data.shape[0] < 1 or min(data.shape[1:]) < _MIN_GRID_SIZE:
            raise ValueError("Spectrum viewer requires at least one plane and a 2 by 2 grid.")
        y_axis, x_axis = spectral_params
        xlim = (float(x_axis.pts2ppm(0)), float(x_axis.pts2ppm(data.shape[-1] - 1)))
        ylim = (float(y_axis.pts2ppm(0)), float(y_axis.pts2ppm(data.shape[-2] - 1)))
        return data, xlim, ylim

    def unalias_y(self, y0: FloatArray) -> FloatArray:
        """Wrap indirect-axis positions using the full spectral width."""
        size = self.data.shape[-2]
        width = (self.ylim[1] - self.ylim[0]) * size / (size - 1)
        return cast("FloatArray", np.asarray((y0 - self.ylim[0]) % width + self.ylim[0]))


class SpectrumViewer:
    """Provide metadata and binary contours without importing Qt or Matplotlib."""

    def __init__(
        self,
        data: NMRData,
        reconstructor: SpectraReconstructor | None = None,
        peaks: pd.DataFrame | None = None,
        *,
        plot_config: PlotConfig | None = None,
    ) -> None:
        self.data = data
        self.reconstructor = reconstructor
        self.peaks = peaks
        self.styles = {}
        if plot_config is not None:
            from peakfit.plot.styles import browser_styles  # noqa: PLC0415

            self.styles = browser_styles(plot_config)
        self._cache: OrderedDict[tuple[int, str, float], bytes] = OrderedDict()
        self._cache_size = 0
        self._simulated: tuple[int, FloatArray] | None = None
        if reconstructor:
            reconstructor.validate_spectrum(data.data.shape, data.xlim, data.ylim)
        sample = data.data.ravel()[:: max(1, data.data.size // _NOISE_SAMPLE_SIZE)]
        sample = sample[np.isfinite(sample)]
        if not sample.size:
            raise ValueError("Spectrum contains no finite intensity values.")
        noise = reconstructor.noise if reconstructor else None
        self.noise = float(noise) if noise and noise > 0 else float(estimate_noise(sample))
        # A noiseless or constant spectrum still needs a positive contour base.
        if not np.isfinite(self.noise) or self.noise <= 0:
            self.noise = max(float(np.max(np.abs(sample))) / 100, np.finfo(float).eps)

    def metadata(self) -> dict[str, Any]:
        """Return axis, layer, and peak information for the browser controls."""
        layers = ["experimental"]
        if self.reconstructor:
            layers.extend(["simulated", "residual"])
        return {
            "filename": Path(self.data.filename).name,
            "shape": self.data.data.shape,
            "xlim": self.data.xlim,
            "ylim": self.data.ylim,
            "x_label": self.data.x_label,
            "y_label": self.data.y_label,
            "noise": self.noise,
            "contour_count": _CONTOUR_COUNT,
            "layers": layers,
            "styles": self.styles,
            "peaks": [] if self.peaks is None else self.peaks.to_dict(orient="records"),
        }

    def plane(self, index: int, layer: str) -> FloatArray:
        """Return an experimental, simulated, or experimental-minus-simulated plane."""
        if not 0 <= index < self.data.data.shape[0]:
            raise ValueError("Plane index is outside the spectrum.")
        experimental = self.data.data[index]
        if layer == "experimental":
            return experimental
        if layer not in {"simulated", "residual"} or self.reconstructor is None:
            raise ValueError("Requested layer is unavailable; supply --results for fit layers.")
        if self._simulated is None or self._simulated[0] != index:
            simulated = self.reconstructor.reconstruct_plane(index, experimental.shape)
            if simulated.shape != experimental.shape or not np.all(np.isfinite(simulated)):
                raise ValueError("Reconstructed spectrum has invalid shape or intensity values.")
            self._simulated = (index, simulated)
        simulated = self._simulated[1]
        return experimental - simulated if layer == "residual" else simulated

    def contours(self, index: int, layer: str, levels: Sequence[float]) -> bytes:
        """Return cached signed paths for explicitly requested heights in noise units.

        The batch starts with a uint32 level count. Each level has a float64
        height, uint32 positive and negative point counts, then float32 XY pairs
        with NaN path breaks. Heights are cached independently of the level set.
        """
        if not 1 <= len(levels) <= _CONTOUR_COUNT or any(
            not np.isfinite(level) or level <= 0 for level in levels
        ):
            raise ValueError("Request between 1 and 25 positive, finite contour heights.")
        if len(set(levels)) != len(levels):
            raise ValueError("Contour heights must be unique.")
        generator = None
        records = []
        separator = np.full((1, 2), np.nan, dtype="<f4").tobytes()
        for level in levels:
            key = (index, layer, level)
            if key in self._cache:
                self._cache.move_to_end(key)
                payload = self._cache[key]
            else:
                if generator is None:
                    generator = contourpy.contour_generator(
                        z=np.ma.masked_invalid(self.plane(index, layer)),
                        line_type="ChunkCombinedNan",
                    )
                sections = []
                for sign in (1, -1):
                    chunks = [
                        np.asarray(points, dtype="<f4").tobytes()
                        for points in generator.lines(float(sign * level * self.noise))[0]
                        if points is not None
                    ]
                    sections.append(separator.join(chunks))
                payload = struct.pack("<II", *(len(section) // 8 for section in sections))
                payload += b"".join(sections)
                if len(payload) <= _CACHE_BYTES:
                    while self._cache and self._cache_size + len(payload) > _CACHE_BYTES:
                        self._cache_size -= len(self._cache.popitem(last=False)[1])
                    self._cache[key] = payload
                    self._cache_size += len(payload)
            records.append(struct.pack("<d", level) + payload)
        return struct.pack("<I", len(levels)) + b"".join(records)


def create_viewer_server(viewer: SpectrumViewer, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Bind a local-only HTTP server with an unguessable session URL."""
    prefix = f"/{secrets.token_urlsafe(24)}/"
    contour_lock = Lock()
    assets = {"": ("viewer.html", "text/html"), "viewer.js": ("viewer.js", "text/javascript")}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Serve bundled assets, metadata, or a contour layer."""
            request = urlsplit(self.path)
            if not request.path.startswith(prefix):
                self.send_error(404)
                return
            route = request.path[len(prefix) :]
            try:
                if route in assets:
                    filename, content_type = assets[route]
                    body = files("peakfit.plot").joinpath(filename).read_bytes()
                elif route == "metadata":
                    body = json.dumps(viewer.metadata(), allow_nan=False).encode()
                    content_type = "application/json"
                elif route == "contours":
                    query = parse_qs(request.query, strict_parsing=True)
                    with contour_lock:
                        body = viewer.contours(
                            int(query["plane"][0]),
                            query["layer"][0],
                            [float(height) for height in query["levels"][0].split(",")],
                        )
                    content_type = "application/octet-stream"
                else:
                    self.send_error(404)
                    return
            except (ValueError, KeyError, IndexError) as error:
                self._reply(400, json.dumps({"error": str(error)}).encode(), "application/json")
                return
            except Exception as error:
                self._reply(500, json.dumps({"error": str(error)}).encode(), "application/json")
                return
            self._reply(200, body, content_type)

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Keep request logging out of the interactive terminal."""

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server, prefix


def serve_viewer(
    data: NMRData,
    reconstructor: SpectraReconstructor | None = None,
    peaks: pd.DataFrame | None = None,
    *,
    port: int = 0,
    open_browser: bool = True,
    on_ready: Callable[[str], None] | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    """Serve the viewer until Ctrl+C; optionally open the system browser."""
    viewer = SpectrumViewer(data, reconstructor, peaks, plot_config=plot_config)
    server, session_path = create_viewer_server(viewer, port)
    with server:
        url = f"http://127.0.0.1:{server.server_port}{session_path}"
        if on_ready:
            on_ready(url)
        if open_browser:
            webbrowser.open(url)
        with suppress(KeyboardInterrupt):
            server.serve_forever(poll_interval=0.2)
