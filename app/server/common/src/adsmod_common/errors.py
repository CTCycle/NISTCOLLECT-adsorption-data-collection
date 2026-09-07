from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

###############################################################################
class ErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    request_id: str
    details: dict[str, str] = Field(default_factory=dict)


__all__ = ["ErrorEnvelope"]
