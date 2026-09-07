from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

###############################################################################
class ParameterConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower: float
    upper: float
    initial: float

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_order(self) -> ParameterConfiguration:
        if self.lower >= self.upper:
            raise ValueError("Parameter lower bound must be smaller than upper bound.")
        if not self.lower <= self.initial <= self.upper:
            raise ValueError("Initial value must lie within the parameter bounds.")
        return self

###############################################################################
class DisplayUnits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pressure: str = "bar"
    uptake: str = "mmol/g"

###############################################################################
class FittingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int = Field(ge=1)
    isotherm_id: int = Field(ge=1)
    models: list[str] = Field(min_length=1, max_length=9)
    optimizer: Literal["trf", "dogbox"] = "trf"
    max_evaluations: int = Field(default=10_000, ge=10, le=1_000_000)
    weighting: Literal["unweighted", "inverse_sigma"] = "unweighted"
    parameter_configuration: dict[str, dict[str, ParameterConfiguration]] = Field(
        default_factory=dict
    )
    display_units: DisplayUnits = Field(default_factory=DisplayUnits)

###############################################################################
class ModelParameterDefinition(BaseModel):
    name: str
    label: str
    lower: float
    upper: float
    initial: float
    unit: str

###############################################################################
class ModelDefinition(BaseModel):
    key: str
    name: str
    equation_latex: str
    assumptions: str
    pressure_requirement: str
    requires_temperature: bool
    reference: str
    parameters: list[ModelParameterDefinition]

###############################################################################
class ModelCatalogResponse(BaseModel):
    status: Literal["success"] = "success"
    pressure_unit: str
    uptake_unit: str
    models: list[ModelDefinition]

###############################################################################
class FittedParameter(BaseModel):
    name: str
    label: str
    value: float
    standard_error: float | None
    ci95_low: float | None
    ci95_high: float | None
    unit: str

###############################################################################
class FitMetrics(BaseModel):
    sse: float | None
    rmse: float | None
    mae: float | None
    r_squared: float | None
    adjusted_r_squared: float | None
    chi_square: float | None
    aic: float | None
    aicc: float | None
    bic: float | None

###############################################################################
class PredictionPoint(BaseModel):
    pressure: float
    observed: float | None = None
    predicted: float
    residual: float | None = None

###############################################################################
class ModelFitResult(BaseModel):
    model: str
    name: str
    status: Literal["success", "warning", "failed"]
    convergence_message: str
    function_evaluations: int | None
    jacobian_rank: int | None
    condition_number: float | None
    parameters: list[FittedParameter]
    metrics: FitMetrics
    observed_predictions: list[PredictionPoint]
    curve: list[PredictionPoint]
    warnings: list[str]
    rank: int | None = None

###############################################################################
class FittingResponse(BaseModel):
    status: Literal["success", "warning", "error"]
    run_id: int | None
    dataset_id: int
    isotherm_id: int
    dataset_name: str
    experiment_name: str
    adsorbent: str
    adsorbate: str
    temperature_k: float
    pressure_basis: str
    pressure_unit: str
    uptake_unit: str
    observation_count: int
    best_model: str | None
    results: list[ModelFitResult]
    summary: str

###############################################################################
class PersistedRunMetricsResponse(BaseModel):
    sse: float | None
    rmse: float | None
    mae: float | None
    r_squared: float | None
    adjusted_r_squared: float | None
    chi_square: float | None
    aic: float | None
    aicc: float | None
    bic: float | None

###############################################################################
class PersistedRunParameterResponse(BaseModel):
    name: str
    value: float
    unit: str
    standard_error: float | None

###############################################################################
class PersistedRunCurvePointResponse(BaseModel):
    pressure: float
    pressure_unit: str
    uptake_mol_kg: float

###############################################################################
class PersistedRunModelResponse(BaseModel):
    id: int
    model: str
    status: str
    convergence_message: str
    metrics: PersistedRunMetricsResponse
    predicted_observations: list[float]
    predicted_curve: list[PersistedRunCurvePointResponse]
    warnings: list[str]
    parameters: list[PersistedRunParameterResponse]

###############################################################################
class PersistedRunResponse(BaseModel):
    status: Literal["success"] = "success"
    run_id: int
    dataset_id: int
    isotherm_id: int
    optimizer: str
    weighting: str
    status_detail: str
    message: str
    created_at: str | None
    completed_at: str | None
    results: list[PersistedRunModelResponse]
