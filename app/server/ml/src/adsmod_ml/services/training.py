from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from adsmod_common.config import AdsmodConfig
from adsmod_common.training_data import TrainingDataAccess
from adsmod_ml.common.utils.logger import logger
from adsmod_ml.contracts.training import (
    CheckpointDetailInfo,
    CheckpointFullDetailsResponse,
    CheckpointsResponse,
    DatasetBuildRequest,
    DatasetInfoResponse,
    DatasetSourceInfo,
    DatasetSourcesResponse,
    OperationStatusResponse,
    ProcessedDatasetInfo,
    ProcessedDatasetsResponse,
    ResumeTrainingRequest,
    TrainingConfigRequest,
    TrainingDatasetResponse,
    TrainingMetadata,
    TrainingStartResponse,
    TrainingStatusResponse,
)
from adsmod_ml.learning.training.manager import TrainingManager, run_training_process
from adsmod_ml.learning.training.worker import ProcessWorker
from adsmod_ml.services.data.builder import DatasetBuilder, DatasetBuilderConfig
from adsmod_ml.services.data.composition import DatasetCompositionService
from adsmod_ml.contracts.jobs import (
    JobCancelResponse,
    JobListResponse,
    JobStartResponse,
    JobStatusResponse,
)
from adsmod_ml.services.job_responses import JobResponseFactory
from adsmod_ml.services.jobs import JobManager

###############################################################################
def get_training_process_stop_timeout_seconds(config: AdsmodConfig) -> float:
    return max(
        5.0,
        float(config.application.jobs.polling_interval),
    )

###############################################################################
class TrainingSession:

    # -------------------------------------------------------------------------
    def __init__(self, training_manager: TrainingManager) -> None:
        self.training_manager = training_manager
        self.worker: ProcessWorker | None = None
        self.current_job_id: str | None = None

    # -------------------------------------------------------------------------
    def reset_for_new_session(
        self,
        total_epochs: int,
        job_id: str,
        current_epoch: int = 0,
        message: str | None = None,
        history: list[dict[str, Any]] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        if history is None:
            history = []
        if metrics is None:
            metrics = {}
        self.training_manager.state.update(
            is_training=True,
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            progress=0.0,
            session_id=job_id,
            stop_requested=False,
            last_error=None,
            metrics=metrics,
            history=history,
            log=[],
        )
        log_message = message or "Starting training session"
        self.training_manager.state.add_log(f"{log_message}: {job_id}")
        self.current_job_id = job_id

    # -------------------------------------------------------------------------
    def finish_session(self) -> None:
        self.worker = None
        self.current_job_id = None

###############################################################################
def determine_checkpoint_compatibility(
    checkpoint_name: str,
    metadata: TrainingMetadata | None,
    dataset_hashes: set[str],
    log_missing_metadata: bool = True,
) -> bool:
    if metadata is None:
        if log_missing_metadata:
            logger.warning(
                "Checkpoint %s metadata missing or invalid; marking incompatible.",
                checkpoint_name,
            )
        return False

    checkpoint_hash = metadata.dataset_hash
    if not checkpoint_hash:
        logger.warning(
            "Checkpoint %s metadata missing dataset_hash; marking incompatible.",
            checkpoint_name,
        )
        return False

    if not dataset_hashes:
        return False

    return checkpoint_hash in dataset_hashes

###############################################################################
class TrainingJobRunner:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        session: TrainingSession,
        job_manager: JobManager,
        training_manager: TrainingManager,
        *,
        config: AdsmodConfig,
        process_context: dict[str, Any],
    ) -> None:
        self.session = session
        self.job_manager = job_manager
        self.training_manager = training_manager
        self.config = config
        self.process_context = dict(process_context)

    # -------------------------------------------------------------------------
    def handle_training_progress(self, job_id: str, message: dict[str, Any]) -> None:
        self.training_manager.handle_process_message(job_id, message)

        if not job_id:
            return

        state = self.training_manager.state.snapshot()
        progress = state.get("progress", 0.0)
        if not isinstance(progress, (int, float)):
            progress = 0.0
        if progress <= 0.0 and state["total_epochs"] > 0:
            progress = (state["current_epoch"] / state["total_epochs"]) * 100

        self.job_manager.update_progress(job_id, progress)
        self.job_manager.update_result(
            job_id,
            {
                "current_epoch": state["current_epoch"],
                "total_epochs": state["total_epochs"],
                "progress": progress,
                "metrics": state["metrics"],
            },
        )

    # -------------------------------------------------------------------------
    def drain_worker_progress(self, job_id: str, worker: ProcessWorker) -> None:
        while True:
            message = worker.poll(timeout=0.0)
            if message is None:
                return
            self.handle_training_progress(job_id, message)

    # -------------------------------------------------------------------------
    def monitor_training_process(
        self,
        job_id: str,
        worker: ProcessWorker,
        stop_timeout_seconds: float,
    ) -> dict[str, Any]:
        stop_requested_at: float | None = None

        while worker.is_alive():
            if self.job_manager.should_stop(job_id):
                if not worker.is_interrupted():
                    worker.stop()
                    stop_requested_at = time.monotonic()
            if stop_requested_at is not None:
                elapsed = time.monotonic() - stop_requested_at
                if elapsed >= stop_timeout_seconds:
                    worker.terminate()
                    break
            message = worker.poll(timeout=0.25)
            if message is not None:
                self.handle_training_progress(job_id, message)
                self.drain_worker_progress(job_id, worker)

        worker.join(timeout=5)
        self.drain_worker_progress(job_id, worker)

        result_payload = worker.read_result()
        if result_payload is None:
            if worker.exitcode not in (0, None) and not self.job_manager.should_stop(
                job_id
            ):
                raise RuntimeError(
                    f"Training process exited with code {worker.exitcode}"
                )
            return {}
        if result_payload.get("error"):
            raise RuntimeError(str(result_payload.get("error")))
        if "result" in result_payload:
            return result_payload.get("result") or {}
        return {}

    # -------------------------------------------------------------------------
    def run_process_job(
        self,
        job_id: str,
        process_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        worker = ProcessWorker()
        self.session.worker = worker
        try:
            worker.start(
                target=run_training_process,
                kwargs={**self.process_context, **process_kwargs},
            )

            result = self.monitor_training_process(
                job_id,
                worker,
                stop_timeout_seconds=get_training_process_stop_timeout_seconds(
                    self.config
                ),
            )

            if self.job_manager.should_stop(job_id):
                self.training_manager.handle_job_completion(
                    job_id, "cancelled", None, None
                )
                return {}

            self.training_manager.handle_job_completion(
                job_id, "completed", result, None
            )
            return result
        except Exception as exc:  # noqa: BLE001
            if self.job_manager.should_stop(job_id):
                self.training_manager.handle_job_completion(
                    job_id, "cancelled", None, None
                )
                return {}
            self.training_manager.handle_job_completion(
                job_id, "failed", None, str(exc)
            )
            raise
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            worker.cleanup()
            self.session.finish_session()

###############################################################################
class TrainingService:
    DATASET_JOB_TYPE = "training_dataset"
    TRAINING_JOB_TYPE = "training"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        config: AdsmodConfig,
        snapshot_access: TrainingDataAccess,
        artifact_root: Path,
        checkpoints_dir: Path,
        job_manager: JobManager,
        training_manager: TrainingManager,
        training_session: TrainingSession,
        training_job_runner: TrainingJobRunner,
    ) -> None:
        self.config = config
        self.snapshot_access = snapshot_access
        self.artifact_root = artifact_root
        self.checkpoints_dir = checkpoints_dir
        self.job_manager = job_manager
        self.training_manager = training_manager
        self.training_session = training_session
        self.training_job_runner = training_job_runner

    # -------------------------------------------------------------------------
    def get_training_datasets(self) -> TrainingDatasetResponse:
        logger.info("Checking training dataset availability")
        processed = self.training_manager.data_serializer.list_processed_datasets()
        info = (
            self.training_manager.data_serializer.get_training_dataset_info(
                str(processed[0]["dataset_label"])
            )
            if processed
            else None
        )
        if info is None:
            return TrainingDatasetResponse(
                available=False,
                name=None,
                train_samples=None,
                validation_samples=None,
            )
        return TrainingDatasetResponse(
            available=True,
            name="Training Dataset",
            train_samples=info.get("train_samples"),
            validation_samples=info.get("validation_samples"),
        )

    # -------------------------------------------------------------------------
    def get_dataset_sources(self) -> DatasetSourcesResponse:
        composer = DatasetCompositionService(self.snapshot_access)
        datasets = [DatasetSourceInfo(**entry) for entry in composer.list_sources()]
        return DatasetSourcesResponse(datasets=datasets)

    # -------------------------------------------------------------------------
    def run_dataset_build(self, request_data: dict[str, Any]) -> dict[str, Any]:
        request = DatasetBuildRequest(**request_data)
        logger.info("Building training dataset with config: %s", request.model_dump())

        reference_metadata = None
        if request.reference_checkpoint:
            try:
                checkpoint_path = (
                    self.training_manager.model_serializer.resolve_checkpoint_path(
                        request.reference_checkpoint
                    )
                )
            except ValueError:
                return {
                    "success": False,
                    "message": "Reference checkpoint name is invalid.",
                }
            if not Path(checkpoint_path).is_dir():
                return {
                    "success": False,
                    "message": "Reference checkpoint not found.",
                }
            try:
                _, reference_metadata, _ = (
                    self.training_manager.model_serializer.load_training_configuration(
                        checkpoint_path
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load reference checkpoint metadata from %s: %s",
                    request.reference_checkpoint,
                    exc,
                )
                return {
                    "success": False,
                    "message": "Failed to load reference checkpoint metadata.",
                }

        config = DatasetBuilderConfig(
            sample_size=request.sample_size,
            validation_size=request.validation_size,
            min_measurements=request.min_measurements,
            max_measurements=request.max_measurements,
            smile_sequence_size=request.smile_sequence_size,
            max_pressure=request.max_pressure,
            max_uptake=request.max_uptake,
        )
        composer = DatasetCompositionService(self.snapshot_access)
        selections = [selection.model_dump() for selection in request.datasets]
        adsorption_data, guest_data, host_data, dataset_label = (
            composer.compose_datasets(selections)
        )

        builder = DatasetBuilder(
            config,
            serializer=self.training_manager.data_serializer,
            dataset_label=request.dataset_label,
        )
        result = builder.build_training_dataset(
            adsorption_data=adsorption_data,
            guest_data=guest_data,
            host_data=host_data,
            dataset_name=dataset_label,
            reference_metadata=reference_metadata,
        )

        if result.get("success"):
            return {
                "success": True,
                "message": "Training dataset built successfully.",
                "total_samples": result.get("total_samples"),
                "train_samples": result.get("train_samples"),
                "validation_samples": result.get("validation_samples"),
            }
        return {
            "success": False,
            "message": result.get("error", "Unknown error during dataset building."),
        }

    # -------------------------------------------------------------------------
    def build_training_dataset(self, request: DatasetBuildRequest) -> JobStartResponse:
        if self.job_manager.is_job_running(self.DATASET_JOB_TYPE):
            raise ValueError("A dataset build job is already running.")

        job_id = self.job_manager.start_job(
            job_type=self.DATASET_JOB_TYPE,
            runner=self.run_dataset_build,
            args=(request.model_dump(),),
        )
        return JobResponseFactory.start(
            job_id=job_id,
            job_type=self.DATASET_JOB_TYPE,
            message="Dataset build job started.",
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def get_dataset_job_status(self, job_id: str) -> JobStatusResponse:
        job_status = self.job_manager.get_job_status(job_id)
        if job_status is None:
            raise LookupError(f"Job {job_id} not found.")
        return JobResponseFactory.status(
            job_status=job_status,
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def list_dataset_jobs(self) -> JobListResponse:
        all_jobs = self.job_manager.list_jobs(self.DATASET_JOB_TYPE)
        return JobResponseFactory.list(
            job_statuses=all_jobs,
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def cancel_dataset_job(self, job_id: str) -> JobCancelResponse:
        success = self.job_manager.cancel_job(job_id)
        if not success:
            raise ValueError(f"Job {job_id} cannot be cancelled.")
        return JobResponseFactory.cancelled(job_id)

    # -------------------------------------------------------------------------
    def get_processed_datasets(self) -> ProcessedDatasetsResponse:
        datasets_list = self.training_manager.data_serializer.list_processed_datasets()
        datasets = [ProcessedDatasetInfo(**entry) for entry in datasets_list]
        return ProcessedDatasetsResponse(datasets=datasets)

    # -------------------------------------------------------------------------
    def get_dataset_info(self, dataset_label: str) -> DatasetInfoResponse:
        resolved_label = self.training_manager.data_serializer.normalize_dataset_label(
            dataset_label
        )
        info = self.training_manager.data_serializer.get_training_dataset_info(
            resolved_label
        )
        if info is None:
            return DatasetInfoResponse(available=False)
        return DatasetInfoResponse(
            available=True,
            dataset_label=info.get("dataset_label"),
            created_at=info.get("created_at"),
            sample_size=info.get("sample_size"),
            validation_size=info.get("validation_size"),
            min_measurements=info.get("min_measurements"),
            max_measurements=info.get("max_measurements"),
            smile_sequence_size=info.get("smile_sequence_size"),
            max_pressure=info.get("max_pressure"),
            max_uptake=info.get("max_uptake"),
            total_samples=info.get("total_samples"),
            train_samples=info.get("train_samples"),
            validation_samples=info.get("validation_samples"),
        )

    # -------------------------------------------------------------------------
    def clear_training_dataset(
        self,
        dataset_label: str | None,
    ) -> OperationStatusResponse:
        resolved_label = (
            self.training_manager.data_serializer.normalize_dataset_label(dataset_label)
            if dataset_label is not None
            else None
        )
        try:
            self.training_manager.data_serializer.clear_training_dataset(resolved_label)
            success = True
        except (OSError, RuntimeError, ValueError):
            success = False
        if success:
            message = (
                f"Training dataset '{resolved_label}' cleared."
                if resolved_label
                else "All training datasets cleared."
            )
            return OperationStatusResponse(status="success", message=message)
        return OperationStatusResponse(
            status="error",
            message="Failed to clear training dataset.",
        )

    # -------------------------------------------------------------------------
    def get_checkpoints(self) -> CheckpointsResponse:
        logger.info("Scanning for available checkpoints")
        checkpoints = self.training_manager.model_serializer.scan_checkpoints_folder()
        dataset_hashes = self.training_manager.data_serializer.collect_dataset_hashes()
        detailed_checkpoints: list[CheckpointDetailInfo] = []

        for checkpoint in checkpoints:
            checkpoint_path = str(self.checkpoints_dir / checkpoint)
            epochs_trained: int | None = None
            final_loss: float | None = None
            final_accuracy: float | None = None
            metadata: TrainingMetadata | None = None
            metadata_load_failed = False

            try:
                training_configuration, metadata, session = (
                    self.training_manager.model_serializer.load_training_configuration(
                        checkpoint_path
                    )
                )
                session_history = (
                    session.get("history") if isinstance(session, dict) else {}
                )
                if not isinstance(session_history, dict):
                    session_history = {}

                epochs_value = (
                    session.get("epochs") if isinstance(session, dict) else None
                )
                if isinstance(epochs_value, int):
                    epochs_trained = epochs_value

                loss_values = session_history.get("loss")
                if isinstance(loss_values, list) and loss_values:
                    last_loss = loss_values[-1]
                    if isinstance(last_loss, (int, float)):
                        final_loss = float(last_loss)
                    if epochs_trained is None:
                        epochs_trained = len(loss_values)

                if epochs_trained is None and isinstance(training_configuration, dict):
                    configured_epochs = training_configuration.get("epochs")
                    if isinstance(configured_epochs, int):
                        epochs_trained = configured_epochs

                for metric_key in [
                    "accuracy",
                    "MaskedR2",
                    "masked_r2",
                    "masked_r_squared",
                    "val_accuracy",
                ]:
                    metric_values = session_history.get(metric_key)
                    if isinstance(metric_values, list) and metric_values:
                        last_value = metric_values[-1]
                        if isinstance(last_value, (int, float)):
                            final_accuracy = float(last_value)
                        break

            except Exception as exc:  # noqa: BLE001
                metadata_load_failed = True
                logger.warning(
                    "Failed to load checkpoint details for %s: %s",
                    checkpoint,
                    exc,
                )

            is_compatible = determine_checkpoint_compatibility(
                checkpoint,
                metadata,
                dataset_hashes,
                log_missing_metadata=not metadata_load_failed,
            )

            detailed_checkpoints.append(
                CheckpointDetailInfo(
                    name=checkpoint,
                    epochs_trained=epochs_trained,
                    final_loss=final_loss,
                    final_accuracy=final_accuracy,
                    is_compatible=is_compatible,
                )
            )

        return CheckpointsResponse(checkpoints=detailed_checkpoints)

    # -------------------------------------------------------------------------
    def get_checkpoint_details(
        self, checkpoint_name: str
    ) -> CheckpointFullDetailsResponse:
        checkpoint_path = (
            self.training_manager.model_serializer.resolve_checkpoint_path(
                checkpoint_name
            )
        )
        if not Path(checkpoint_path).is_dir():
            raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found")

        configuration, metadata, history = (
            self.training_manager.model_serializer.load_training_configuration(
                checkpoint_path
            )
        )
        return CheckpointFullDetailsResponse(
            name=checkpoint_name,
            configuration=configuration,
            metadata=metadata,
            history=history,
        )

    # -------------------------------------------------------------------------
    def delete_checkpoint(self, checkpoint_name: str) -> OperationStatusResponse:
        success = self.training_manager.model_serializer.delete_checkpoint(
            checkpoint_name
        )
        if not success:
            raise ValueError(f"Failed to delete checkpoint {checkpoint_name}.")
        return OperationStatusResponse(
            status="success",
            message=f"Checkpoint {checkpoint_name} deleted.",
        )

    # -------------------------------------------------------------------------
    def start_training(self, config: TrainingConfigRequest) -> TrainingStartResponse:
        state = self.training_manager.state.snapshot()
        if state["is_training"] or self.job_manager.is_job_running(
            self.TRAINING_JOB_TYPE
        ):
            raise ValueError("Training is already in progress. Stop it first.")

        resolved_label = self.training_manager.data_serializer.normalize_dataset_label(
            config.dataset_label
        )
        configuration = config.model_dump()
        configuration["dataset_label"] = resolved_label
        configuration["polling_interval"] = (
            self.config.application.jobs.polling_interval
        )

        logger.info("Starting training with config: %s", configuration)
        metadata = self.training_manager.data_serializer.load_training_metadata(
            resolved_label
        )
        requested_hash = configuration.get("dataset_hash")
        if (
            requested_hash
            and metadata.dataset_hash
            and requested_hash != metadata.dataset_hash
        ):
            raise ValueError(
                "Selected dataset does not match the stored training metadata. "
                "Refresh the dataset list and try again."
            )
        info = self.training_manager.data_serializer.get_training_dataset_info(
            resolved_label
        )
        if info is None:
            raise ValueError("No training dataset available. Build the dataset first.")

        job_id = self.job_manager.start_job(
            job_type=self.TRAINING_JOB_TYPE,
            runner=self.training_job_runner.run_process_job,
            kwargs={
                "process_kwargs": {"configuration": configuration},
            },
        )

        total_epochs = int(configuration.get("epochs", 0))
        self.training_session.reset_for_new_session(
            total_epochs=total_epochs,
            job_id=job_id,
            current_epoch=0,
            message="Starting training session",
        )
        state_snapshot = self.training_manager.state.snapshot()
        self.job_manager.update_result(
            job_id,
            {
                "current_epoch": state_snapshot["current_epoch"],
                "total_epochs": state_snapshot["total_epochs"],
                "progress": 0.0,
                "metrics": state_snapshot["metrics"],
            },
        )

        return TrainingStartResponse(
            status="started",
            session_id=job_id,
            message=f"Training started with {config.epochs} epochs. Session: {job_id}",
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def resume_training(self, request: ResumeTrainingRequest) -> TrainingStartResponse:
        state = self.training_manager.state.snapshot()
        if state["is_training"] or self.job_manager.is_job_running(
            self.TRAINING_JOB_TYPE
        ):
            raise ValueError("Training is already in progress. Stop it first.")

        logger.info(
            "Resuming training from checkpoint: %s with %s additional epochs",
            request.checkpoint_name,
            request.additional_epochs,
        )
        available = self.training_manager.model_serializer.scan_checkpoints_folder()
        if request.checkpoint_name not in available:
            raise FileNotFoundError(
                f"Checkpoint '{request.checkpoint_name}' not found."
            )

        checkpoint_path = (
            self.training_manager.model_serializer.resolve_checkpoint_path(
                request.checkpoint_name
            )
        )
        _, _, session = (
            self.training_manager.model_serializer.load_training_configuration(
                checkpoint_path
            )
        )

        from_epoch = 0
        if isinstance(session, dict):
            from_epoch_value = session.get("epochs", 0)
            if isinstance(from_epoch_value, int):
                from_epoch = from_epoch_value
        history_entries = self.training_manager.build_history_entries(
            session if isinstance(session, dict) else {}
        )
        last_metrics = self.training_manager.extract_last_metrics(history_entries)

        job_id = self.job_manager.start_job(
            job_type=self.TRAINING_JOB_TYPE,
            runner=self.training_job_runner.run_process_job,
            kwargs={
                "process_kwargs": {
                    "configuration": None,
                    "checkpoint": request.checkpoint_name,
                    "additional_epochs": request.additional_epochs,
                },
            },
        )

        total_epochs = from_epoch + request.additional_epochs
        self.training_session.reset_for_new_session(
            total_epochs=total_epochs,
            job_id=job_id,
            current_epoch=from_epoch,
            message="Resuming training session",
            history=history_entries,
            metrics=last_metrics,
        )
        state_snapshot = self.training_manager.state.snapshot()
        self.job_manager.update_result(
            job_id,
            {
                "current_epoch": state_snapshot["current_epoch"],
                "total_epochs": state_snapshot["total_epochs"],
                "progress": (from_epoch / total_epochs * 100.0)
                if total_epochs
                else 0.0,
                "metrics": state_snapshot["metrics"],
            },
        )

        return TrainingStartResponse(
            status="started",
            session_id=job_id,
            message=(
                f"Resuming training from {request.checkpoint_name} "
                f"with {request.additional_epochs} epochs. Session: {job_id}"
            ),
            poll_interval=self.config.application.jobs.polling_interval,
        )

    # -------------------------------------------------------------------------
    def stop_training(self) -> OperationStatusResponse:
        state = self.training_manager.state.snapshot()
        if not state["is_training"]:
            return OperationStatusResponse(
                status="stopped",
                message="No training session is running.",
            )

        logger.info("Stop requested for current training session")
        self.training_manager.state.update(stop_requested=True)
        self.training_manager.state.add_log("Stop requested by user...")
        if self.training_session.worker is not None:
            self.training_session.worker.stop()
        if self.training_session.current_job_id:
            self.job_manager.cancel_job(self.training_session.current_job_id)

        return OperationStatusResponse(
            status="stopped",
            message="Training stop requested.",
        )

    # -------------------------------------------------------------------------
    def get_training_status(self) -> TrainingStatusResponse:
        state = self.training_manager.state.snapshot()
        progress = state.get("progress", 0.0)
        if not isinstance(progress, (int, float)):
            progress = 0.0
        if progress <= 0.0 and state["total_epochs"] > 0:
            progress = (state["current_epoch"] / state["total_epochs"]) * 100

        return TrainingStatusResponse(
            is_training=state["is_training"],
            current_epoch=state["current_epoch"],
            total_epochs=state["total_epochs"],
            progress=progress,
            metrics=state["metrics"],
            history=state["history"],
            log=state["log"],
            poll_interval=self.config.application.jobs.polling_interval,
        )
