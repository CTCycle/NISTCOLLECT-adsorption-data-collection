from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

###############################################################################
class NumericBounds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: int | float
    maximum: int | float

###############################################################################
class ParameterDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float
    upper: float
    initial: float

###############################################################################
class DisplayUnitCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pressure: tuple[str, ...]
    uptake: tuple[str, ...]
    default_pressure: str
    default_uptake: str

###############################################################################
class FittingConfigurationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success"] = "success"
    supported_optimizers: tuple[str, ...]
    default_optimizer: str
    default_max_evaluations: int
    max_evaluations_bounds: NumericBounds
    weighting_options: tuple[str, ...]
    default_weighting: str
    display_units: DisplayUnitCapabilities
    parameter_defaults: ParameterDefaults


__all__ = [
    "DisplayUnitCapabilities",
    "FittingConfigurationResponse",
    "NumericBounds",
    "ParameterDefaults",
]
