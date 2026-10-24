"""Public headless Python interface. See docs/python-api.md for its contract."""

from peakfit.engine.domain.config import PeakFitConfig
from peakfit.engine.domain.window import ProcessingWindow
from peakfit.fit.api import (
    FitCancelledError,
    FitOptions,
    ModelAvailability,
    PeakEstimate,
    PeakSeed,
    PreparedFit,
    SeedDecision,
    SpectrumFitResult,
    fit_file,
    fit_spectrum,
    model_availability,
    prepare_spectrum,
)
from peakfit.io.spectrum_input import (
    AxisExtraction,
    AxisProcessing,
    ExtractedSpectrum,
    Extraction,
    ResourceEstimate,
    SeriesAxis,
    SpectrumInput,
    extract_spectrum,
    plan_extraction,
)

__all__ = [
    "AxisExtraction",
    "AxisProcessing",
    "ExtractedSpectrum",
    "Extraction",
    "FitCancelledError",
    "FitOptions",
    "ModelAvailability",
    "PeakEstimate",
    "PeakFitConfig",
    "PeakSeed",
    "PreparedFit",
    "ProcessingWindow",
    "ResourceEstimate",
    "SeedDecision",
    "SeriesAxis",
    "SpectrumFitResult",
    "SpectrumInput",
    "extract_spectrum",
    "fit_file",
    "fit_spectrum",
    "model_availability",
    "plan_extraction",
    "prepare_spectrum",
]
