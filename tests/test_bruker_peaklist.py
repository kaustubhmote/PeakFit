from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from typer.testing import CliRunner

from peakfit.cli.app import app
from peakfit.engine.domain.config import FitConfig
from peakfit.fit.validation import ValidationResult, _load_peak_rows, _validate_peaklist
from peakfit.io.readers.peaks import read_bruker_peak_rows, read_list

if TYPE_CHECKING:
    from pathlib import Path

    from peakfit.engine.domain.spectrum import Spectra

XML = """<?xml version="1.0" encoding="UTF-8"?>
<PeakList>
  <PeakList2D>
    <PeakList2DHeader><PeakPickDetails>Manually picked</PeakPickDetails></PeakList2DHeader>
    <Peak2D F1="110.20655310524418" F2="6.798254769363926" intensity="76264.5" type="0"/>
  </PeakList2D>
  <PeakList2D>
    <Peak2D F1="106.044825986512" F2="7.687335110198672" intensity="-506935" type="1"/>
  </PeakList2D>
</PeakList>
"""


def test_bruker_rows_and_peak_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "peaklist.xml"
    path.write_text(XML)
    rows = read_bruker_peak_rows(path)
    assert rows == [
        ("peak1", [110.20655310524418, 6.798254769363926]),
        ("peak2", [106.044825986512, 7.687335110198672]),
    ]
    assert _load_peak_rows(path) == rows
    spectra = cast("Spectra", SimpleNamespace(n_spectral_dims=2))
    monkeypatch.setattr(
        "peakfit.io.readers.peaks.create_shapes",
        lambda *_args: [SimpleNamespace(), SimpleNamespace()],
    )
    peaks = read_list(path, spectra, ["gaussian", "gaussian"], FitConfig())
    assert [p.name for p in peaks] == ["peak1", "peak2"]
    np.testing.assert_array_equal(peaks[0].positions, rows[0][1])
    np.testing.assert_array_equal(peaks[1].positions, rows[1][1])
    with pytest.raises(ValueError, match="exactly two spectral dimensions"):
        read_list(path, cast("Spectra", SimpleNamespace(n_spectral_dims=1)), [], FitConfig())


@pytest.mark.parametrize(
    ("xml", "message"),
    [
        ("<PeakList>", "Invalid Bruker peak list XML"),
        ("<other/>", "Expected Bruker PeakList"),
        ("<PeakList/>", "No Peak2D entries"),
        (XML.replace('F1="110.20655310524418"', ""), "Invalid F1 position"),
        (XML.replace('F2="6.798254769363926"', 'F2="bad"'), "Invalid F2 position"),
        (XML.replace('F2="6.798254769363926"', 'F2="nan"'), "Invalid F2 position"),
        (XML.replace('F2="6.798254769363926"', 'F2="inf"'), "Invalid F2 position"),
    ],
)
def test_invalid_xml_is_a_validation_error(tmp_path: Path, xml: str, message: str) -> None:
    path = tmp_path / "peaklist.xml"
    path.write_text(xml)
    result = ValidationResult()
    _validate_peaklist(path, result)
    assert len(result.errors) == 1
    assert message in result.errors[0]


@pytest.mark.parametrize("input_form", ["binary", "processed", "experiment"])
@pytest.mark.parametrize("explicit", [False, True])
def test_cli_discovers_bruker_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_form: str, explicit: bool
) -> None:
    processed = tmp_path / "pdata" / "1"
    processed.mkdir(parents=True)
    (processed / "procs").touch()
    binary = processed / "2rr"
    binary.touch()
    discovered = processed / "peaklist.xml"
    discovered.write_text(XML)
    supplied = tmp_path / "peaks.csv"
    supplied.write_text("name,F1_ppm,F2_ppm\nA,110,6.8\n")
    spectrum = {"binary": binary, "processed": processed, "experiment": tmp_path}[input_form]
    expected = supplied if explicit else discovered
    captured = {}
    monkeypatch.setattr("peakfit.fit.validation._validate_spectrum", lambda *_args: None)

    def load(*, peaklist: Path | None, **_kwargs: object) -> object:
        captured["loaded"] = peaklist
        return SimpleNamespace(peaks=[])

    def run(_data: object, _config: object, output_dir: Path, **_kwargs: object) -> object:
        return SimpleNamespace(output_dir=output_dir, spectra=SimpleNamespace())

    def write(*args: object) -> None:
        captured["inputs"] = args[3]

    monkeypatch.setattr("peakfit.cli.commands.fit._load_fit_data", load)
    monkeypatch.setattr("peakfit.cli.commands.fit.run_fit", run)
    monkeypatch.setattr("peakfit.cli.commands.fit.write_fit_run_outputs", write)
    args = ["fit", str(spectrum)]
    if explicit:
        args.append(str(supplied))
    result = CliRunner().invoke(app, [*args, "--headless", "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert captured["loaded"] == expected
    assert cast("dict[str, Path]", captured["inputs"])["peaklist"] == expected
