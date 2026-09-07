from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from adsmod_ml.contracts.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)

###############################################################################
class JobResponseFactory:

    # -------------------------------------------------------------------------
    @staticmethod
    def start(
        job_id: str,
        job_type: str,
        message: str,
        poll_interval: float | None,
    ) -> JobStartResponse:
        return JobStartResponse(
            job_id=job_id,
            job_type=job_type,
            status="running",
            message=message,
            poll_interval=poll_interval,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def status(
        job_status: Mapping[str, Any],
        poll_interval: float | None,
    ) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=str(job_status["job_id"]),
            job_type=str(job_status["job_type"]),
            status=str(job_status["status"]),
            progress=float(job_status["progress"]),
            result=job_status.get("result"),
            error=job_status.get("error"),
            poll_interval=poll_interval,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def list(
        cls,
        job_statuses: list[Mapping[str, Any]],
        poll_interval: float | None,
    ) -> JobListResponse:
        return JobListResponse(
            jobs=[cls.status(job_status, poll_interval) for job_status in job_statuses]
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def cancelled(job_id: str) -> JobCancelResponse:
        return JobCancelResponse(status="cancelled", job_id=job_id)
