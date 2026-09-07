from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

###############################################################################
class RuntimeDeviceCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    keras_backend: str
    cuda_available: bool
    device_count: int
    devices: tuple[str, ...]

###############################################################################
class TrainingConfigurationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success"] = "success"
    defaults: dict[str, Any]
    dataset_defaults: dict[str, Any]
    resume_defaults: dict[str, Any]
    numeric_constraints: dict[str, dict[str, int | float]]
    supported_models: tuple[str, ...]
    checkpoint_capabilities: dict[str, bool]
    runtime: RuntimeDeviceCapabilities


__all__ = ["RuntimeDeviceCapabilities", "TrainingConfigurationResponse"]
