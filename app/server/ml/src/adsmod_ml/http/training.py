from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, status

from adsmod_ml.contracts.training import (
    CheckpointFullDetailsResponse,
    CheckpointsResponse,
    DatasetBuildRequest,
    DatasetInfoResponse,
    DatasetSourcesResponse,
    OperationStatusResponse,
    ProcessedDatasetsResponse,
    ResumeTrainingRequest,
    TrainingConfigRequest,
    TrainingDatasetResponse,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from adsmod_ml.common.utils.logger import logger
from adsmod_ml.services.container import MlServiceContainer
from adsmod_ml.services.training import TrainingService
from adsmod_ml.contracts.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)

###############################################################################
class TrainingEndpoint:

    # -------------------------------------------------------------------------
    def __init__(self, router: APIRouter, service: TrainingService) -> None:
        self.router = router
        self.service = service

    # -------------------------------------------------------------------------
    def get_training_datasets(self) -> TrainingDatasetResponse:
        try:
            return self.service.get_training_datasets()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error checking training datasets: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to check training dataset availability.",
            ) from exc

    # -------------------------------------------------------------------------
    def get_dataset_sources(self) -> DatasetSourcesResponse:
        try:
            return self.service.get_dataset_sources()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error listing dataset sources: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to list dataset sources.",
            ) from exc

    # -------------------------------------------------------------------------
    def build_training_dataset(self, request: DatasetBuildRequest) -> JobStartResponse:
        try:
            return self.service.build_training_dataset(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Error starting dataset build: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to start dataset build.",
            ) from exc

    # -------------------------------------------------------------------------
    def get_dataset_job_status(self, job_id: str) -> JobStatusResponse:
        try:
            return self.service.get_dataset_job_status(job_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Error getting dataset job status: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to get dataset job status.",
            ) from exc

    # -------------------------------------------------------------------------
    def list_dataset_jobs(self) -> JobListResponse:
        try:
            return self.service.list_dataset_jobs()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error listing dataset jobs: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to list dataset jobs.",
            ) from exc

    # -------------------------------------------------------------------------
    def cancel_dataset_job(self, job_id: str) -> JobCancelResponse:
        try:
            return self.service.cancel_dataset_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Error cancelling dataset job: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to cancel dataset job.",
            ) from exc

    # -------------------------------------------------------------------------
    def get_processed_datasets(self) -> ProcessedDatasetsResponse:
        try:
            return self.service.get_processed_datasets()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error listing processed datasets: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to list processed datasets.",
            ) from exc

    # -------------------------------------------------------------------------
    def get_dataset_info(
        self,
        dataset_label: str = Query(
            "default",
            min_length=1,
            max_length=64,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$",
        ),
    ) -> DatasetInfoResponse:
        try:
            return self.service.get_dataset_info(dataset_label)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error getting dataset info: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to load dataset info.",
            ) from exc

    # -------------------------------------------------------------------------
    def clear_training_dataset(
        self,
        dataset_label: str | None = Query(
            default=None,
            min_length=1,
            max_length=64,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$",
        ),
    ) -> OperationStatusResponse:
        try:
            return self.service.clear_training_dataset(dataset_label)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error clearing training dataset: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to clear training dataset.",
            ) from exc

    # -------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
        try:
            return self.service.get_checkpoints()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error scanning checkpoints: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to scan checkpoints.",
            ) from exc

    # -------------------------------------------------------------------------
    def get_checkpoint_details(
        self,
        checkpoint_name: str = Path(
            ...,
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
        ),
    ) -> CheckpointFullDetailsResponse:
        try:
            return self.service.get_checkpoint_details(checkpoint_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Error getting checkpoint details: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to load checkpoint details.",
            ) from exc

    # -------------------------------------------------------------------------
    def delete_checkpoint(
        self,
        checkpoint_name: str = Path(
            ...,
            min_length=1,
            max_length=128,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
        ),
    ) -> OperationStatusResponse:
        try:
            return self.service.delete_checkpoint(checkpoint_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Error deleting checkpoint: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to delete checkpoint.",
            ) from exc

    # -------------------------------------------------------------------------
    def start_training(self, config: TrainingConfigRequest) -> TrainingStartResponse:
        try:
            return self.service.start_training(config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Error starting training: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to start training.",
            ) from exc

    # -------------------------------------------------------------------------
    def resume_training(self, request: ResumeTrainingRequest) -> TrainingStartResponse:
        try:
            return self.service.resume_training(request)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Error resuming training: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to resume training.",
            ) from exc

    # -------------------------------------------------------------------------
    def stop_training(self) -> OperationStatusResponse:
        try:
            return self.service.stop_training()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error stopping training: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to stop training.",
            ) from exc

    # -------------------------------------------------------------------------
    def get_training_status(self) -> TrainingStatusResponse:
        try:
            return self.service.get_training_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("Error getting training status: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to get training status.",
            ) from exc

    # -------------------------------------------------------------------------
    def add_routes(self) -> None:
        self.router.add_api_route(
            "/datasets",
            self.get_training_datasets,
            methods=["GET"],
            response_model=TrainingDatasetResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset-sources",
            self.get_dataset_sources,
            methods=["GET"],
            response_model=DatasetSourcesResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/build-dataset",
            self.build_training_dataset,
            methods=["POST"],
            response_model=JobStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/processed-datasets",
            self.get_processed_datasets,
            methods=["GET"],
            response_model=ProcessedDatasetsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset-info",
            self.get_dataset_info,
            methods=["GET"],
            response_model=DatasetInfoResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/dataset",
            self.clear_training_dataset,
            methods=["DELETE"],
            response_model=OperationStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs",
            self.list_dataset_jobs,
            methods=["GET"],
            response_model=JobListResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.get_dataset_job_status,
            methods=["GET"],
            response_model=JobStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/jobs/{job_id}",
            self.cancel_dataset_job,
            methods=["DELETE"],
            response_model=JobCancelResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/checkpoints",
            self.get_checkpoints,
            methods=["GET"],
            response_model=CheckpointsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/checkpoints/{checkpoint_name}",
            self.get_checkpoint_details,
            methods=["GET"],
            response_model=CheckpointFullDetailsResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/checkpoints/{checkpoint_name}",
            self.delete_checkpoint,
            methods=["DELETE"],
            response_model=OperationStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/start",
            self.start_training,
            methods=["POST"],
            response_model=TrainingStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/resume",
            self.resume_training,
            methods=["POST"],
            response_model=TrainingStartResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/stop",
            self.stop_training,
            methods=["POST"],
            response_model=OperationStatusResponse,
            status_code=status.HTTP_200_OK,
        )
        self.router.add_api_route(
            "/status",
            self.get_training_status,
            methods=["GET"],
            response_model=TrainingStatusResponse,
            status_code=status.HTTP_200_OK,
        )

###############################################################################
def create_training_router(container: MlServiceContainer) -> APIRouter:
    router = APIRouter(prefix="/training", tags=["training"])
    endpoint = TrainingEndpoint(router=router, service=container.training_service)
    endpoint.add_routes()
    return router
