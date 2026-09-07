from __future__ import annotations

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)

from adsmod_core.contracts.datasets import (
    DatasetImportResponse,
    DatasetListResponse,
    DatasetMetadata,
    DatasetMutationResponse,
    DatasetRenameRequest,
    ExperimentListResponse,
    ImportPreviewResponse,
    ImportValidationResponse,
    ObservationPage,
    SupportedUnitsResponse,
)
from adsmod_core.services.container import CoreServiceContainer
from adsmod_core.services.data.datasets import DatasetService
from adsmod_core.common.constants import DATASETS_ROUTER_PREFIX, MAX_UPLOAD_SIZE_BYTES

###############################################################################
class DatasetEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: DatasetService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    @staticmethod
    def bad_request(exc: ValueError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    def not_found(exc: LookupError) -> HTTPException:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    # -------------------------------------------------------------------------
    @staticmethod
    async def read_upload_bytes(file: UploadFile) -> bytes:
        payload = bytearray()
        try:
            while chunk := await file.read(1024 * 1024):
                payload.extend(chunk)
                if len(payload) > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Uploaded dataset exceeds the 25 MB limit.",
                    )
        finally:
            await file.close()
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded dataset is empty.",
            )
        return bytes(payload)

    # -------------------------------------------------------------------------
    async def preview_import(
        self, file: UploadFile = File(...)
    ) -> ImportPreviewResponse:
        try:
            payload = await self.read_upload_bytes(file)
            return self.service.preview(payload, file.filename)
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    async def validate_import(
        self,
        mapping: str = Form(...),
        file: UploadFile = File(...),
    ) -> ImportValidationResponse:
        try:
            payload = await self.read_upload_bytes(file)
            return self.service.validate(
                payload, file.filename, self.service.parse_mapping(mapping)
            )
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    async def commit_import(
        self,
        mapping: str = Form(...),
        file: UploadFile = File(...),
    ) -> DatasetImportResponse:
        try:
            payload = await self.read_upload_bytes(file)
            return self.service.commit(
                payload, file.filename, self.service.parse_mapping(mapping)
            )
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def list_datasets(self) -> DatasetListResponse:
        return self.service.list_datasets()

    # -------------------------------------------------------------------------
    def supported_units(self) -> SupportedUnitsResponse:
        return self.service.supported_units()

    # -------------------------------------------------------------------------
    def list_experiments(
        self, dataset_id: int = Path(..., ge=1)
    ) -> ExperimentListResponse:
        try:
            return self.service.list_experiments(dataset_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def get_observations(
        self,
        dataset_id: int = Path(..., ge=1),
        isotherm_id: int = Path(..., ge=1),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ) -> ObservationPage:
        try:
            return self.service.get_observations(dataset_id, isotherm_id, offset, limit)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def rename_dataset(
        self,
        request: DatasetRenameRequest,
        dataset_id: int = Path(..., ge=1),
    ) -> DatasetMutationResponse:
        try:
            return self.service.rename(dataset_id, request.new_name)
        except LookupError as exc:
            raise self.not_found(exc) from exc
        except ValueError as exc:
            raise self.bad_request(exc) from exc

    # -------------------------------------------------------------------------
    def update_metadata(
        self,
        request: DatasetMetadata,
        dataset_id: int = Path(..., ge=1),
    ) -> DatasetMutationResponse:
        try:
            return self.service.update_metadata(dataset_id, request)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def delete_dataset(self, dataset_id: int = Path(..., ge=1)) -> None:
        try:
            self.service.delete(dataset_id)
        except LookupError as exc:
            raise self.not_found(exc) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/supported-units",
            self.supported_units,
            methods=["GET"],
            response_model=SupportedUnitsResponse,
        )
        self.router.add_api_route(
            "/import/preview",
            self.preview_import,
            methods=["POST"],
            response_model=ImportPreviewResponse,
        )
        self.router.add_api_route(
            "/import/validate",
            self.validate_import,
            methods=["POST"],
            response_model=ImportValidationResponse,
        )
        self.router.add_api_route(
            "/import/commit",
            self.commit_import,
            methods=["POST"],
            response_model=DatasetImportResponse,
            status_code=status.HTTP_201_CREATED,
        )
        self.router.add_api_route(
            "",
            self.list_datasets,
            methods=["GET"],
            response_model=DatasetListResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}/experiments",
            self.list_experiments,
            methods=["GET"],
            response_model=ExperimentListResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}/experiments/{isotherm_id}/observations",
            self.get_observations,
            methods=["GET"],
            response_model=ObservationPage,
        )
        self.router.add_api_route(
            "/{dataset_id}/rename",
            self.rename_dataset,
            methods=["PATCH"],
            response_model=DatasetMutationResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}/metadata",
            self.update_metadata,
            methods=["PATCH"],
            response_model=DatasetMutationResponse,
        )
        self.router.add_api_route(
            "/{dataset_id}",
            self.delete_dataset,
            methods=["DELETE"],
            status_code=status.HTTP_204_NO_CONTENT,
        )

###############################################################################
def create_dataset_router(container: CoreServiceContainer) -> APIRouter:
    router = APIRouter(prefix=DATASETS_ROUTER_PREFIX, tags=["datasets"])
    DatasetEndpoint(router, container.dataset_service).add_routes()
    return router
