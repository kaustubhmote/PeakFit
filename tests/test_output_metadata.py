from __future__ import annotations

from typing import TYPE_CHECKING

from peakfit.fit.output_metadata import capture_output_metadata

if TYPE_CHECKING:
    from pathlib import Path


def test_metadata_hashes_directory_inputs_deterministically(tmp_path: Path) -> None:
    spectrum = tmp_path / "pdata" / "1"
    spectrum.mkdir(parents=True)
    (spectrum / "procs").write_text("parameters", encoding="utf-8")
    (spectrum / "2rr").write_bytes(b"spectrum")

    first = capture_output_metadata({}, {"spectrum": spectrum})
    second = capture_output_metadata({}, {"spectrum": spectrum})

    assert first.input_files["spectrum"]["path"] == str(spectrum.resolve())
    assert (
        first.input_files["spectrum"]["checksum_sha256"]
        == second.input_files["spectrum"]["checksum_sha256"]
    )
