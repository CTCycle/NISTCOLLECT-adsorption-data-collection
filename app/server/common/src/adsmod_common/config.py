from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

###############################################################################
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

###############################################################################
class RuntimeConfig(StrictModel):
    host: str
    backend_port: int = Field(ge=1024, le=65535)
    frontend_port: int = Field(ge=1024, le=65535)

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_runtime(self) -> "RuntimeConfig":
        if self.backend_port == self.frontend_port:
            raise ValueError("runtime.backend_port and runtime.frontend_port must be distinct")
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("runtime.host must be a loopback address")
        return self

###############################################################################
class StorageConfig(StrictModel):
    root: Path

###############################################################################
class DatabaseConfig(StrictModel):
    embedded_database: bool
    engine: str = "postgres"
    host: str | None = None
    port: int = Field(default=5432, ge=1, le=65535)
    database_name: str | None = None
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    ssl_ca: str | None = None
    connect_timeout: int = Field(default=30, ge=1)
    insert_batch_size: int = Field(default=5000, ge=1)
    sqlite_path: str | None

    # -------------------------------------------------------------------------
    @field_validator(
        "host",
        "database_name",
        "username",
        "password",
        "ssl_ca",
        "sqlite_path",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    @field_validator("engine", mode="before")
    @classmethod
    def normalize_engine(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "postgres"

###############################################################################
class DatasetConfig(StrictModel):
    allowed_extensions: tuple[str, ...]
    column_detection_cutoff: float = Field(ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def normalize_extensions(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return (".csv", ".xls", ".xlsx")
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item) for item in value]
        else:
            raise ValueError("datasets.allowed_extensions must be a sequence or string")

        cleaned = tuple(part.strip() for part in values if str(part).strip())
        if not cleaned:
            raise ValueError("datasets.allowed_extensions must not be empty")
        return cleaned

###############################################################################
class NISTConfig(StrictModel):
    parallel_tasks: int = Field(ge=1)
    pubchem_parallel_tasks: int = Field(ge=1, le=3)

###############################################################################
class PublicDataConfig(StrictModel):
    request_timeout_seconds: float = Field(default=20.0, gt=0.0, le=120.0)
    retry_attempts: int = Field(default=3, ge=1, le=6)
    pubchem_parallel_requests: int = Field(default=2, ge=1, le=3)
    cod_max_interactive_results: int = Field(default=250, ge=1, le=1000)

###############################################################################
class FittingConfig(StrictModel):
    default_max_iterations: int = Field(ge=1)
    max_iterations_upper_bound: int = Field(ge=1)
    default_parameter_initial: float = Field(ge=0.0)
    default_parameter_min: float = Field(ge=0.0)
    default_parameter_max: float = Field(ge=0.0)
    preview_row_limit: int = Field(ge=1)
    best_model_metric: str

    # -------------------------------------------------------------------------
    @field_validator("best_model_metric", mode="before")
    @classmethod
    def normalize_metric(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "AICc"

    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_bounds(self) -> "FittingConfig":
        if self.max_iterations_upper_bound < self.default_max_iterations:
            raise ValueError(
                "fitting.max_iterations_upper_bound must be >= fitting.default_max_iterations"
            )
        if self.default_parameter_max < self.default_parameter_min:
            raise ValueError(
                "fitting.default_parameter_max must be >= fitting.default_parameter_min"
            )
        return self

###############################################################################
class JobConfig(StrictModel):
    polling_interval: float = Field(gt=0.0)

###############################################################################
class TrainingConfig(StrictModel):
    use_jit: bool = False
    jit_backend: str = "inductor"
    use_mixed_precision: bool = False
    dataloader_workers: int = Field(default=0, ge=0)
    persistent_workers: bool

    # -------------------------------------------------------------------------
    @field_validator("jit_backend", mode="before")
    @classmethod
    def normalize_backend(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        return text or "inductor"

###############################################################################
class ApplicationConfig(StrictModel):
    database: DatabaseConfig
    datasets: DatasetConfig
    nist: NISTConfig
    public_data: PublicDataConfig
    fitting: FittingConfig
    jobs: JobConfig
    training: TrainingConfig

###############################################################################
class AdsmodConfig(StrictModel):
    version: Literal["3.0.0"]
    runtime: RuntimeConfig
    storage: StorageConfig
    application: ApplicationConfig

###############################################################################
def load_config(path: str | Path) -> AdsmodConfig:
    config_path = Path(path)
    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"configuration is not valid JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a JSON object")
    return AdsmodConfig.model_validate(payload)


__all__ = [
    "AdsmodConfig",
    "ApplicationConfig",
    "DatabaseConfig",
    "DatasetConfig",
    "FittingConfig",
    "JobConfig",
    "NISTConfig",
    "PublicDataConfig",
    "RuntimeConfig",
    "StorageConfig",
    "TrainingConfig",
    "load_config",
]
