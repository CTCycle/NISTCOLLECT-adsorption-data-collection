from __future__ import annotations

from typing import Any

from pathlib import Path

from adsmod_common.config import AdsmodConfig
from adsmod_common.paths import resolve_checkpoint_root, resolve_storage_root
from adsmod_common.training_data import TrainingDataAccess
from adsmod_ml.learning.callbacks import WorkerInterrupted
from adsmod_ml.learning.device import DeviceConfig
from adsmod_ml.learning.loader import (
    SCADSAtomicDataLoader,
    SCADSDataLoader,
)
from adsmod_ml.learning.models.qmodel import SCADSAtomicModel, SCADSModel
from adsmod_ml.learning.serialization.model import ModelSerializer
from adsmod_ml.learning.serialization.training import TrainingDataSerializer
from adsmod_ml.learning.training.fitting import ModelTraining
from adsmod_ml.common.utils.logger import logger
from adsmod_ml.learning.training.state import TrainingState
from adsmod_ml.common.constants import SCADS_ATOMIC_MODEL, SCADS_SERIES_MODEL


MODEL_COMPONENTS = {
    SCADS_SERIES_MODEL: (SCADSModel, SCADSDataLoader),
    SCADS_ATOMIC_MODEL: (SCADSAtomicModel, SCADSAtomicDataLoader),
}

HISTORY_KEY_ALIASES = {
    "MaskedAccuracy": "accuracy",
    "val_MaskedAccuracy": "val_accuracy",
    "masked_accuracy": "accuracy",
    "val_masked_accuracy": "val_accuracy",
    "MaskedR2": "masked_r2",
    "val_MaskedR2": "val_masked_r2",
    "masked_r_squared": "masked_r2",
    "val_masked_r_squared": "val_masked_r2",
}

###############################################################################
def put_worker_result(result_queue: Any | None, payload: dict[str, Any]) -> None:
    if result_queue is None:
        return
    try:
        result_queue.put(payload, block=False)
    except Exception:
        try:
            result_queue.put(payload)
        except Exception:
            return

###############################################################################
class TrainingProcessRunner:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        worker: Any | None = None,
        *,
        snapshot_access: TrainingDataAccess,
        artifact_root: Path,
        checkpoints_dir: Path,
    ) -> None:
        self.worker = worker
        self.data_serializer = TrainingDataSerializer(snapshot_access, artifact_root)
        self.model_serializer = ModelSerializer(checkpoints_dir)

    # -------------------------------------------------------------------------
    def should_stop(self) -> bool:
        if self.worker is None:
            return False
        checker = getattr(self.worker, "is_interrupted", None)
        if callable(checker):
            return bool(checker())
        return False

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, total_epochs: int, logs: dict[str, Any]) -> None:
        self.send_training_message(
            self.worker,
            {
                "type": "epoch_end",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "logs": logs,
            },
        )

    # -------------------------------------------------------------------------
    def log(self, message: str) -> None:
        self.send_training_message(
            self.worker,
            {
                "type": "log",
                "message": message,
            },
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_model_name(name: str | None) -> str:
        if not name:
            return SCADS_SERIES_MODEL
        lowered = name.strip().lower()
        if "atomic" in lowered:
            return SCADS_ATOMIC_MODEL
        return SCADS_SERIES_MODEL

    # -------------------------------------------------------------------------
    @staticmethod
    def send_training_message(worker: Any | None, payload: dict[str, Any]) -> None:
        if worker is None:
            return
        sender = getattr(worker, "send_message", None)
        if not callable(sender):
            return
        try:
            sender(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send training message: %s", exc)

    # -------------------------------------------------------------------------
    def ensure_required_columns(self, data: Any, required: list[str]) -> None:
        if data is None or getattr(data, "empty", True):
            raise ValueError("Training dataset is empty.")
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Training dataset missing columns: {', '.join(missing)}")

    # -------------------------------------------------------------------------
    def validate_resume_model(self, model: Any) -> None:
        optimizer = getattr(model, "optimizer", None)
        loss = getattr(model, "loss", None)
        if optimizer is None or loss is None:
            raise ValueError(
                "Checkpoint model is not compiled or missing optimizer state. "
                "Resume requires a compiled model with saved optimizer momentum."
            )
        opt_vars: list[Any] = []
        variables = getattr(optimizer, "variables", None)
        if variables is not None:
            if callable(variables):
                opt_vars = list(variables())
            else:
                opt_vars = list(variables)
        if not opt_vars:
            get_weights = getattr(optimizer, "get_weights", None)
            if callable(get_weights):
                opt_vars = list(get_weights())
        if not opt_vars:
            raise ValueError(
                "Checkpoint optimizer state is empty. Resume requires optimizer "
                "momentum to be saved in the checkpoint."
            )

    # -------------------------------------------------------------------------
    def start_training(self, configuration: dict[str, Any]) -> None:
        dataset_label = self.data_serializer.normalize_dataset_label(
            configuration.get("dataset_label")
        )
        train_data, validation_data, metadata = self.data_serializer.load_training_data(
            dataset_label
        )
        if train_data.empty or validation_data.empty:
            raise ValueError("No training data available. Build the dataset first.")

        selected_model = self.normalize_model_name(configuration.get("selected_model"))
        model_builder, dataloader_builder = MODEL_COMPONENTS.get(
            selected_model, MODEL_COMPONENTS[SCADS_SERIES_MODEL]
        )

        required_columns = [
            "temperature",
            "pressure",
            "adsorbed_amount",
            "adsorbate_encoded_SMILE",
            "adsorbate_molecular_weight",
            "encoded_adsorbent",
        ]
        self.ensure_required_columns(train_data, required_columns)
        self.ensure_required_columns(validation_data, required_columns)

        train_loader = dataloader_builder(
            configuration,
            metadata.model_dump(),
            shuffle=configuration.get("shuffle_dataset", True),
        )
        val_loader = dataloader_builder(
            configuration, metadata.model_dump(), shuffle=False
        )
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(configuration).set_device()
        custom_name = configuration.get("custom_name")
        if custom_name and isinstance(custom_name, str) and custom_name.strip():
            safe_name = "".join(
                c for c in custom_name.strip() if c.isalnum() or c in ("-", "_")
            )
            model_name = safe_name if safe_name else selected_model.replace(" ", "_")
        else:
            model_name = selected_model.replace(" ", "_")

        self.model_serializer = ModelSerializer(
            self.model_serializer.checkpoints_dir,
            model_name=model_name,
        )
        checkpoint_path = self.model_serializer.create_checkpoint_folder()

        wrapper = model_builder(configuration, metadata.model_dump())
        model = wrapper.get_model(model_summary=True)

        trainer = ModelTraining(configuration, metadata.model_dump())
        model, history = trainer.train_model(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            should_stop=self.should_stop,
            on_epoch_end=self.on_epoch_end,
            worker=self.worker,
        )

        self.model_serializer.save_pretrained_model(model, checkpoint_path)
        self.model_serializer.save_training_configuration(
            checkpoint_path, history, configuration, metadata
        )

    # -------------------------------------------------------------------------
    def resume_training(self, checkpoint: str, additional_epochs: int) -> None:
        (
            model,
            train_config,
            model_metadata,
            session,
            checkpoint_path,
        ) = self.model_serializer.load_checkpoint(checkpoint)
        self.validate_resume_model(model)

        dataset_label = self.data_serializer.normalize_dataset_label(
            train_config.get("dataset_label")
        )
        current_metadata = self.data_serializer.load_training_metadata(dataset_label)
        if not self.data_serializer.validate_metadata(current_metadata, model_metadata):
            raise ValueError(
                "Training dataset metadata does not match the checkpoint. "
                "Rebuild the dataset using the checkpoint configuration before resuming."
            )

        train_data, validation_data, _ = self.data_serializer.load_training_data(
            dataset_label
        )
        if train_data.empty or validation_data.empty:
            raise ValueError("No training data available. Build the dataset first.")

        selected_model = self.normalize_model_name(train_config.get("selected_model"))
        _, dataloader_builder = MODEL_COMPONENTS.get(
            selected_model, MODEL_COMPONENTS[SCADS_SERIES_MODEL]
        )

        required_columns = [
            "temperature",
            "pressure",
            "adsorbed_amount",
            "adsorbate_encoded_SMILE",
            "adsorbate_molecular_weight",
            "encoded_adsorbent",
        ]
        self.ensure_required_columns(train_data, required_columns)
        self.ensure_required_columns(validation_data, required_columns)

        train_loader = dataloader_builder(
            train_config,
            model_metadata.model_dump(),
            shuffle=train_config.get("shuffle_dataset", True),
        )
        val_loader = dataloader_builder(
            train_config, model_metadata.model_dump(), shuffle=False
        )
        train_dataset = train_loader.build_training_dataloader(train_data)
        validation_dataset = val_loader.build_training_dataloader(validation_data)

        DeviceConfig(train_config).set_device()

        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs
        self.send_training_message(
            self.worker,
            {
                "type": "state_update",
                "current_epoch": from_epoch,
                "total_epochs": total_epochs,
            },
        )

        trainer = ModelTraining(train_config, model_metadata.model_dump())
        model, history = trainer.resume_training(
            model,
            train_dataset,
            validation_dataset,
            checkpoint_path,
            session,
            additional_epochs,
            should_stop=self.should_stop,
            on_epoch_end=self.on_epoch_end,
            worker=self.worker,
        )

        self.model_serializer.save_pretrained_model(model, checkpoint_path)
        self.model_serializer.save_training_configuration(
            checkpoint_path, history, train_config, model_metadata
        )

###############################################################################
def run_training_process(
    configuration: dict[str, Any] | None,
    checkpoint: str | None = None,
    additional_epochs: int = 0,
    worker: Any | None = None,
    config_payload: dict[str, Any] | None = None,
    artifact_root: str = "",
    checkpoints_dir: str = "",
) -> None:
    result_queue = getattr(worker, "result_queue", None)
    stop_event = getattr(worker, "stop_event", None)
    snapshot_access = None
    try:
        if stop_event is not None and stop_event.is_set():
            put_worker_result(result_queue, {"result": {}})
            return
        if config_payload is None or not artifact_root or not checkpoints_dir:
            raise ValueError("ML process runtime configuration and paths are required.")
        from adsmod_core.services.training_data import open_training_data_service
        runtime_config = AdsmodConfig.model_validate(config_payload)
        snapshot_access = open_training_data_service(runtime_config)
        runner = TrainingProcessRunner(worker=worker, snapshot_access=snapshot_access, artifact_root=Path(artifact_root), checkpoints_dir=Path(checkpoints_dir))
        if checkpoint:
            runner.log(f"Resuming training from checkpoint {checkpoint} for {additional_epochs} additional epochs.")
            runner.resume_training(checkpoint, additional_epochs)
            put_worker_result(result_queue, {"result": {"success": True, "checkpoint": checkpoint}})
            return
        if configuration is None:
            raise ValueError("Training configuration is required.")
        runner.log("Starting training session.")
        runner.start_training(configuration)
        put_worker_result(result_queue, {"result": {"success": True}})
    except WorkerInterrupted:
        put_worker_result(result_queue, {"result": {}})
    except Exception as exc:  # noqa: BLE001
        put_worker_result(result_queue, {"error": str(exc)})
    finally:
        if snapshot_access is not None:
            snapshot_access.close()

###############################################################################
class TrainingManager:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        config: AdsmodConfig,
        *,
        snapshot_access: TrainingDataAccess,
        artifact_root: Path | None = None,
        checkpoints_dir: Path | None = None,
    ) -> None:
        resolved_artifact_root = artifact_root or resolve_storage_root(config) / "training"
        resolved_checkpoints_dir = checkpoints_dir or resolve_checkpoint_root(config)
        self.state = TrainingState()
        self.data_serializer = TrainingDataSerializer(snapshot_access, resolved_artifact_root)
        self.model_serializer = ModelSerializer(resolved_checkpoints_dir)

    # -------------------------------------------------------------------------
    def build_history_entries(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(session, dict):
            return []
        session_history = session.get("history")
        if not isinstance(session_history, dict):
            return []
        lengths = [
            len(values)
            for values in session_history.values()
            if isinstance(values, list)
        ]
        if not lengths:
            return []
        max_len = max(lengths)
        entries: list[dict[str, Any]] = []
        for index in range(max_len):
            entry: dict[str, Any] = {"epoch": index + 1}
            for key, values in session_history.items():
                if not isinstance(values, list) or index >= len(values):
                    continue
                value = values[index]
                if isinstance(value, (int, float)):
                    entry[HISTORY_KEY_ALIASES.get(key, key)] = float(value)
            entries.append(entry)
        return entries

    # -------------------------------------------------------------------------
    def extract_last_metrics(
        self, history_entries: list[dict[str, Any]]
    ) -> dict[str, float]:
        if not history_entries:
            return {}
        last_entry = history_entries[-1]
        metrics: dict[str, float] = {}
        for key, value in last_entry.items():
            if key == "epoch":
                continue
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics

    # -------------------------------------------------------------------------
    def handle_process_message(self, job_id: str, message: dict[str, Any]) -> None:
        if job_id != self.state.session_id:
            return

        message_type = message.get("type")
        if message_type == "epoch_end":
            epoch = message.get("epoch")
            total_epochs = message.get("total_epochs")
            logs = message.get("logs")
            if (
                isinstance(epoch, int)
                and isinstance(total_epochs, int)
                and isinstance(logs, dict)
            ):
                self._on_epoch_end(epoch, total_epochs, logs)
            return

        if message_type == "state_update":
            current_epoch = message.get("current_epoch")
            total_epochs = message.get("total_epochs")
            update_payload: dict[str, Any] = {}
            if isinstance(current_epoch, int):
                update_payload["current_epoch"] = current_epoch
            if isinstance(total_epochs, int):
                update_payload["total_epochs"] = total_epochs
            if update_payload:
                self.state.update(**update_payload)
            return

        if message_type == "training_update":
            current_epoch = message.get("epoch")
            total_epochs = message.get("total_epochs")
            progress_percent = message.get("progress_percent")
            update_payload: dict[str, Any] = {}
            if isinstance(current_epoch, int):
                update_payload["current_epoch"] = current_epoch
            if isinstance(total_epochs, int):
                update_payload["total_epochs"] = total_epochs
            if isinstance(progress_percent, (int, float)):
                update_payload["progress"] = float(progress_percent)
            metrics_update: dict[str, float] = {}
            for key in [
                "loss",
                "val_loss",
                "accuracy",
                "val_accuracy",
                "masked_r2",
                "val_masked_r2",
            ]:
                value = message.get(key)
                if isinstance(value, (int, float)):
                    metrics_update[key] = float(value)
            if metrics_update:
                current_metrics = self.state.snapshot().get("metrics", {})
                if isinstance(current_metrics, dict):
                    merged = dict(current_metrics)
                    merged.update(metrics_update)
                    update_payload["metrics"] = merged
                else:
                    update_payload["metrics"] = metrics_update
            if update_payload:
                self.state.update(**update_payload)
            return

        if message_type == "log":
            message_text = message.get("message")
            if message_text:
                self.state.add_log(str(message_text))
            return

        if message_type == "error":
            error_text = message.get("error")
            if error_text:
                self.state.update(last_error=str(error_text))
                self.state.add_log(f"Training error: {error_text}")

    # -------------------------------------------------------------------------
    def handle_job_completion(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
    ) -> None:
        if job_id != self.state.session_id:
            return

        if status == "failed":
            self.state.update(last_error=error)
            message = error or "Training failed."
            self.state.add_log(f"Training failed: {message}")
        elif status == "cancelled":
            self.state.add_log("Training cancelled.")
        elif status == "completed":
            self.state.add_log("Training completed.")
        elif status:
            self.state.add_log(f"Training finished with status: {status}")

        completion_payload = {"is_training": False, "stop_requested": False}
        if status == "completed":
            completion_payload["progress"] = 100.0
        self.state.update(**completion_payload)

    # -------------------------------------------------------------------------
    def _on_epoch_end(
        self, epoch: int, total_epochs: int, logs: dict[str, Any]
    ) -> None:
        progress = 0.0
        if total_epochs > 0:
            progress = (epoch / total_epochs) * 100
        self.state.update(
            current_epoch=epoch, total_epochs=total_epochs, progress=progress
        )

        # Extract metrics
        loss_value = logs.get("loss")
        loss = float(loss_value) if isinstance(loss_value, (int, float)) else 0.0
        val_loss_value = logs.get("val_loss")
        val_loss = (
            float(val_loss_value) if isinstance(val_loss_value, (int, float)) else 0.0
        )

        accuracy_value = None
        for key in ["accuracy", "MaskedAccuracy", "masked_accuracy"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                accuracy_value = candidate
                break
        accuracy = (
            float(accuracy_value) if isinstance(accuracy_value, (int, float)) else 0.0
        )

        val_accuracy_value = None
        for key in ["val_accuracy", "val_MaskedAccuracy", "val_masked_accuracy"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                val_accuracy_value = candidate
                break
        val_accuracy = (
            float(val_accuracy_value)
            if isinstance(val_accuracy_value, (int, float))
            else 0.0
        )

        masked_r2_value = None
        for key in ["MaskedR2", "masked_r2", "masked_r_squared"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                masked_r2_value = candidate
                break
        masked_r2 = (
            float(masked_r2_value) if isinstance(masked_r2_value, (int, float)) else 0.0
        )

        val_masked_r2_value = None
        for key in ["val_MaskedR2", "val_masked_r2", "val_masked_r_squared"]:
            candidate = logs.get(key)
            if isinstance(candidate, (int, float)):
                val_masked_r2_value = candidate
                break
        val_masked_r2 = (
            float(val_masked_r2_value)
            if isinstance(val_masked_r2_value, (int, float))
            else 0.0
        )

        metrics = {
            "loss": loss,
            "val_loss": val_loss,
        }
        if isinstance(accuracy_value, (int, float)):
            metrics["accuracy"] = accuracy
        if isinstance(val_accuracy_value, (int, float)):
            metrics["val_accuracy"] = val_accuracy
        if isinstance(masked_r2_value, (int, float)):
            metrics["masked_r2"] = masked_r2
        if isinstance(val_masked_r2_value, (int, float)):
            metrics["val_masked_r2"] = val_masked_r2
        self.state.update(metrics=metrics)

        metric_label = "acc" if isinstance(accuracy_value, (int, float)) else "r2"
        metric_value = (
            accuracy if isinstance(accuracy_value, (int, float)) else masked_r2
        )
        val_metric_value = (
            val_accuracy
            if isinstance(val_accuracy_value, (int, float))
            else val_masked_r2
        )

        # Add generic log entry
        log_message = (
            f"Epoch {epoch}/{total_epochs} - loss: {loss:.4f} - "
            f"{metric_label}: {metric_value:.4f} - val_loss: {val_loss:.4f} - "
            f"val_{metric_label}: {val_metric_value:.4f}"
        )
        self.state.add_log(log_message)

        # Add to history for plotting
        history_entry = {"epoch": epoch, **metrics}
        self.state.add_history(history_entry)
