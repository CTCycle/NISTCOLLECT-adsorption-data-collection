from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NISTCategory = Literal["experiments", "guest", "host"]

###############################################################################
class NISTFetchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiments_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    guest_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    host_fraction: float = Field(default=1.0, ge=0.0, le=1.0)

###############################################################################
class NISTFetchResponse(BaseModel):
    status: str = Field(default="success")
    experiments_count: int
    single_component_rows: int
    binary_mixture_rows: int
    skipped_experiment_count: int = 0
    guest_rows: int
    host_rows: int

###############################################################################
class NISTPropertiesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["guest", "host"] = Field(default="guest")

###############################################################################
class NISTPropertiesResponse(BaseModel):
    status: str = Field(default="success")
    target: str
    names_requested: int
    names_matched: int
    rows_updated: int

###############################################################################
class NISTStatusResponse(BaseModel):
    status: str = Field(default="success")
    data_available: bool
    single_component_rows: int
    binary_mixture_rows: int
    guest_rows: int
    host_rows: int

###############################################################################
class NISTCategoryFetchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fraction: float = Field(default=1.0, ge=0.001, le=1.0)

###############################################################################
class NISTCategoryStatus(BaseModel):
    category: NISTCategory
    local_count: int
    available_count: int
    last_update: str | None = None
    server_ok: bool | None = None
    server_checked_at: str | None = None
    supports_enrichment: bool

###############################################################################
class NISTCategoryStatusResponse(BaseModel):
    status: str = Field(default="success")
    categories: list[NISTCategoryStatus]

###############################################################################
class NISTCategoryPingResponse(BaseModel):
    status: str = Field(default="success")
    category: NISTCategory
    server_ok: bool
    checked_at: str

###############################################################################
class NISTCategoryOperationResponse(BaseModel):
    status: str = Field(default="success")
    category: NISTCategory
    available_count: int | None = None
    local_count: int | None = None
    requested_count: int | None = None
    fetched_count: int | None = None
    skipped_count: int | None = None
    names_requested: int | None = None
    names_matched: int | None = None
    rows_updated: int | None = None
