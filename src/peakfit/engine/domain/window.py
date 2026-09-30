"""Fixed acquisition sampling and apodization for one spectral axis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from peakfit.shared.typing import FloatArray

WindowKind = Literal["none", "em", "gm", "gmb", "sine"]


class ProcessingWindow(BaseModel):
    """A recorded window, independent of extracted spectral coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    kind: WindowKind
    points: int = Field(ge=2)
    dwell: float = Field(gt=0)
    duration: float = Field(gt=0, description="Duration used to define the window, seconds.")
    q1: float = 0.0
    q2: float = 0.0
    q3: float = 0.0
    first_point: float = Field(default=0.5, ge=0, le=2)

    def samples(self) -> FloatArray:
        """Return the applied window at t[n]=n*dwell, including first-point scaling."""
        t = np.arange(self.points, dtype=float) * self.dwell
        if self.kind == "em":
            values = np.exp(-np.pi * self.q1 * t)
        elif self.kind == "gm":
            values = np.exp(
                np.pi * self.q1 * t - (0.6 * np.pi * self.q2 * (self.q3 * self.duration - t)) ** 2
            )
        elif self.kind == "gmb":
            b = -np.pi * self.q1 / (2 * self.q2 * self.duration) if self.q2 else 0.0
            values = np.exp(-np.pi * self.q1 * t - b * t**2)
        elif self.kind == "sine":
            angle = np.pi * (self.q1 + (self.q2 - self.q1) * t / self.duration)
            values = np.maximum(np.sin(angle), 0.0) ** self.q3
        else:
            values = np.ones(self.points)
        values[0] *= self.first_point
        if not np.all(np.isfinite(values)):
            raise ValueError("Window overflows at the recorded acquisition duration.")
        return values
