"""adsmod_core Pydantic transport/workflow contracts for background jobs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

###############################################################################
class JobStartResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    message: str
    poll_interval: float | None = None

###############################################################################
class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float
    result: dict[str, Any] | None = None
    error: str | None = None
    poll_interval: float | None = None

###############################################################################
class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]

###############################################################################
class JobCancelResponse(BaseModel):
    status: str
    job_id: str

###############################################################################
class StatusMessageResponse(BaseModel):
    status: str
    message: str
