from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from adsmod_core.contracts.nist import (
    NISTCategory,
    NISTCategoryFetchRequest,
    NISTCategoryPingResponse,
    NISTCategoryStatusResponse,
    NISTFetchRequest,
    NISTPropertiesRequest,
    NISTStatusResponse,
)
from adsmod_core.common.utils.logger import logger
from adsmod_core.services.container import CoreServiceContainer
from adsmod_core.services.data.nist_service import NISTDataService
from adsmod_core.common.constants import (
    NIST_CATEGORY_ENRICH_ENDPOINT,
    NIST_CATEGORY_FETCH_ENDPOINT,
    NIST_CATEGORY_INDEX_ENDPOINT,
    NIST_CATEGORY_PING_ENDPOINT,
    NIST_CATEGORY_STATUS_ENDPOINT,
    NIST_FETCH_ENDPOINT,
    NIST_JOBS_ENDPOINT,
    NIST_JOB_STATUS_ENDPOINT,
    NIST_PROPERTIES_ENDPOINT,
    NIST_ROUTER_PREFIX,
    NIST_STATUS_ENDPOINT,
)
from adsmod_core.contracts.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)

###############################################################################
class NistEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: NISTDataService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def start_fetch_job(self, request: NISTFetchRequest) -> JobStartResponse:
        try:
            response = self.service.start_fetch_job(request)
            logger.info("Started NIST fetch job %s", response.job_id)
            return response
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def start_properties_job(self, request: NISTPropertiesRequest) -> JobStartResponse:
        try:
            response = self.service.start_properties_job(request)
            logger.info(
                "Started NIST properties job %s (target=%s)",
                response.job_id,
                request.target,
            )
            return response
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    async def ping_category_server(
        self, category: NISTCategory
    ) -> NISTCategoryPingResponse:
        try:
            if category == "experiments":
                result = await self.service.ping_experiments_server()
            elif category == "guest":
                result = await self.service.ping_guest_server()
            else:
                result = await self.service.ping_host_server()
        except Exception as exc:  # noqa: BLE001
            logger.exception("NIST category ping failed (category=%s)", category)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to ping NIST category server.",
            ) from exc

        return NISTCategoryPingResponse(
            status="success",
            category=category,
            server_ok=bool(result.get("server_ok", False)),
            checked_at=str(result.get("checked_at", "")),
        )

    # -------------------------------------------------------------------------
    def start_category_index_job(self, category: NISTCategory) -> JobStartResponse:
        try:
            response = self.service.start_category_index_job(category)
            logger.info("Started NIST %s index job %s", category, response.job_id)
            return response
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def start_category_fetch_job(
        self, category: NISTCategory, request: NISTCategoryFetchRequest
    ) -> JobStartResponse:
        try:
            response = self.service.start_category_fetch_job(category, request)
            logger.info("Started NIST %s fetch job %s", category, response.job_id)
            return response
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    def start_category_enrich_job(self, category: NISTCategory) -> JobStartResponse:
        try:
            response = self.service.start_category_enrich_job(category)
            logger.info("Started NIST %s enrichment job %s", category, response.job_id)
            return response
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    # -------------------------------------------------------------------------
    async def fetch_nist_category_status(self) -> NISTCategoryStatusResponse:
        try:
            categories = await self.service.get_category_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("NIST category status check failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load NIST category status.",
            ) from exc

        return NISTCategoryStatusResponse(status="success", categories=categories)

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
    async def fetch_nist_status(self) -> NISTStatusResponse:
        try:
            result = await self.service.get_status()
        except Exception as exc:  # noqa: BLE001
            logger.exception("NIST status check failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load NIST status.",
            ) from exc

        return NISTStatusResponse(
            status="success",
            data_available=bool(result.get("data_available", False)),
            single_component_rows=int(result.get("single_component_rows", 0)),
            binary_mixture_rows=int(result.get("binary_mixture_rows", 0)),
            guest_rows=int(result.get("guest_rows", 0)),
            host_rows=int(result.get("host_rows", 0)),
        )

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            NIST_FETCH_ENDPOINT,
            self.start_fetch_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_PROPERTIES_ENDPOINT,
            self.start_properties_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_STATUS_ENDPOINT,
            self.fetch_nist_status,
            methods=["GET"],
            response_model=NISTStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_STATUS_ENDPOINT,
            self.fetch_nist_category_status,
            methods=["GET"],
            response_model=NISTCategoryStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_PING_ENDPOINT,
            self.ping_category_server,
            methods=["POST"],
            response_model=NISTCategoryPingResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_INDEX_ENDPOINT,
            self.start_category_index_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_FETCH_ENDPOINT,
            self.start_category_fetch_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_CATEGORY_ENRICH_ENDPOINT,
            self.start_category_enrich_job,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_JOBS_ENDPOINT,
            self.list_jobs,
            methods=["GET"],
            response_model=JobListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_JOB_STATUS_ENDPOINT,
            self.get_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            NIST_JOB_STATUS_ENDPOINT,
            self.cancel_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def create_nist_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix=NIST_ROUTER_PREFIX, tags=["nist"])
    endpoint = NistEndpoint(router=router, service=container.nist_service)
    endpoint.add_routes()
    return router
