from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from adsmod_core.contracts.fitting import (
    FittingRequest,
    ModelCatalogResponse,
    PersistedRunResponse,
)
from adsmod_core.services.container import CoreServiceContainer
from adsmod_core.services.fitting import FittingService
from adsmod_core.common.constants import (
    FITTING_JOBS_ENDPOINT,
    FITTING_JOB_STATUS_ENDPOINT,
    FITTING_ROUTER_PREFIX,
    FITTING_RUN_ENDPOINT,
)
from adsmod_core.contracts.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)

###############################################################################
class FittingEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: FittingService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def start_fitting_job(self, payload: FittingRequest) -> JobStartResponse:
        try:
            return self.service.start_fitting_job(payload)
        except (ValueError, LookupError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def model_catalog(
        self,
        pressure_unit: str = "bar",
        uptake_unit: str = "mmol/g",
        dataset_id: int | None = None,
        isotherm_id: int | None = None,
    ) -> ModelCatalogResponse:
        try:
            return self.service.model_catalog(
                pressure_unit, uptake_unit, dataset_id, isotherm_id
            )
        except (ValueError, LookupError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def get_job_status(self, job_id: str) -> JobStatusResponse:
        try:
            return self.service.get_job_status(job_id)
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def list_jobs(self) -> JobListResponse:
        return self.service.list_jobs()

    # -------------------------------------------------------------------------
    def cancel_job(self, job_id: str) -> JobCancelResponse:
        try:
            return self.service.cancel_job(job_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def get_persisted_run(self, run_id: int) -> PersistedRunResponse:
        try:
            return self.service.get_persisted_run(run_id)
        except LookupError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/models",
            self.model_catalog,
            methods=["GET"],
            response_model=ModelCatalogResponse,
        )
        self.router.add_api_route(
            FITTING_RUN_ENDPOINT,
            self.start_fitting_job,
            methods=["POST"],
            response_model=JobStartResponse,
        )
        self.router.add_api_route(
            FITTING_JOBS_ENDPOINT,
            self.list_jobs,
            methods=["GET"],
            response_model=JobListResponse,
        )
        self.router.add_api_route(
            FITTING_JOB_STATUS_ENDPOINT,
            self.get_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
        )
        self.router.add_api_route(
            FITTING_JOB_STATUS_ENDPOINT,
            self.cancel_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
        )
        self.router.add_api_route(
            "/runs/{run_id}",
            self.get_persisted_run,
            methods=["GET"],
            response_model=PersistedRunResponse,
        )

###############################################################################
def create_fitting_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix=FITTING_ROUTER_PREFIX, tags=["fitting"])
    FittingEndpoint(router=router, service=container.fitting_service).add_routes()
    return router
