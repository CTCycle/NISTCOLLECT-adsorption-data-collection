from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HealthState = Literal["starting", "ready", "not-ready", "failed", "unavailable"]

###############################################################################
class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    service: Literal["backend"]
    version: str
    state: HealthState
    details: dict[str, str] = Field(default_factory=dict)


__all__ = ["HealthResponse", "HealthState"]
