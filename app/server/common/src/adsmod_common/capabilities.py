from __future__ import annotations

from pydantic import BaseModel


###############################################################################
class FeatureCapabilities(BaseModel):
    datasets: bool
    nist: bool
    fitting: bool
    machine_learning: bool
    training: bool
    checkpoints: bool


###############################################################################
class CapabilitiesResponse(BaseModel):
    version: str
    features: FeatureCapabilities


__all__ = ["CapabilitiesResponse", "FeatureCapabilities"]
