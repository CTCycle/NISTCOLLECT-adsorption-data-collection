from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keras
from keras.callbacks import Callback

from adsmod_ml.common.utils.logger import logger

###############################################################################
class WorkerInterrupted(RuntimeError):
    """Raised to immediately interrupt training in a worker process."""

###############################################################################
class WorkerProgressMessenger:

    # -------------------------------------------------------------------------
    def __init__(self, worker: Any | None = None) -> None:
        self.worker = worker

    # -------------------------------------------------------------------------
    def __call__(self, payload: dict[str, Any]) -> None:
        if self.worker is None:
            return
        sender = getattr(self.worker, "send_message", None)
        if not callable(sender):
            return
        try:
            sender(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to send training progress message: %s", exc)


# [CALLBACK FOR TRAINING PROGRESS]

###############################################################################
class TrainingProgressCallback(Callback):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        total_epochs: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        on_epoch_end: Callable[[int, int, dict[str, Any]], None] | None = None,
        start_epoch: int = 0,
        polling_interval: float = 1.0,
    ) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_callback = progress_callback
        self.on_epoch_end_callback = on_epoch_end
        self.start_epoch = start_epoch
        self.polling_interval = polling_interval
        self.last_update_time = 0.0
        self.start_time = time.time()
        self.current_epoch_index = start_epoch
        self.last_val_loss = 0.0
        self.last_val_accuracy = 0.0
        self.last_val_masked_r2 = 0.0

    # -------------------------------------------------------------------------
    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self.current_epoch_index = epoch

    # -------------------------------------------------------------------------
    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        current_time = time.time()
        if current_time - self.last_update_time < self.polling_interval:
            return

        self.last_update_time = current_time
        logs = logs or {}

        steps_per_epoch = int(self.params.get("steps", 1) if self.params else 1)
        epoch_progress = (batch + 1) / steps_per_epoch
        total_epochs_to_run = max(1, self.total_epochs - self.start_epoch)
        completed_epochs = self.current_epoch_index - self.start_epoch
        progress_percent = int(
            100 * (completed_epochs + epoch_progress) / total_epochs_to_run
        )
        elapsed_time = current_time - self.start_time

        train_loss = float(logs.get("loss", 0.0))
        accuracy_value = logs.get("MaskedAccuracy", logs.get("accuracy", None))
        train_accuracy = (
            float(accuracy_value) if isinstance(accuracy_value, (int, float)) else None
        )
        masked_r2_value = logs.get("MaskedR2", logs.get("masked_r2", None))
        train_masked_r2 = (
            float(masked_r2_value)
            if isinstance(masked_r2_value, (int, float))
            else None
        )

        message: dict[str, Any] = {
            "type": "training_update",
            "epoch": self.current_epoch_index + 1,
            "total_epochs": self.total_epochs,
            "progress_percent": progress_percent,
            "elapsed_seconds": int(elapsed_time),
            "loss": train_loss,
            "val_loss": self.last_val_loss,
        }
        if train_accuracy is not None:
            message["accuracy"] = train_accuracy
            message["val_accuracy"] = self.last_val_accuracy
        if train_masked_r2 is not None:
            message["masked_r2"] = train_masked_r2
            message["val_masked_r2"] = self.last_val_masked_r2

        if self.progress_callback is not None:
            self.progress_callback(message)

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        logs = logs or {}
        self.last_val_loss = float(logs.get("val_loss", 0.0))

        val_accuracy_value = logs.get(
            "val_MaskedAccuracy", logs.get("val_accuracy", None)
        )
        if isinstance(val_accuracy_value, (int, float)):
            self.last_val_accuracy = float(val_accuracy_value)

        val_masked_r2_value = logs.get("val_MaskedR2", logs.get("val_masked_r2", None))
        if isinstance(val_masked_r2_value, (int, float)):
            self.last_val_masked_r2 = float(val_masked_r2_value)

        current_time = time.time()
        self.last_update_time = current_time
        processed_epochs = epoch - self.start_epoch + 1
        total_epochs_to_run = max(1, self.total_epochs - self.start_epoch)
        progress_percent = int(100 * processed_epochs / total_epochs_to_run)
        elapsed_time = current_time - self.start_time

        message: dict[str, Any] = {
            "type": "training_update",
            "epoch": epoch + 1,
            "total_epochs": self.total_epochs,
            "progress_percent": progress_percent,
            "elapsed_seconds": int(elapsed_time),
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": self.last_val_loss,
        }

        accuracy_value = logs.get("MaskedAccuracy", logs.get("accuracy", None))
        if isinstance(accuracy_value, (int, float)):
            message["accuracy"] = float(accuracy_value)
            message["val_accuracy"] = self.last_val_accuracy

        masked_r2_value = logs.get("MaskedR2", logs.get("masked_r2", None))
        if isinstance(masked_r2_value, (int, float)):
            message["masked_r2"] = float(masked_r2_value)
            message["val_masked_r2"] = self.last_val_masked_r2

        if self.progress_callback is not None:
            self.progress_callback(message)

        if self.on_epoch_end_callback is not None:
            self.on_epoch_end_callback(epoch + 1, self.total_epochs, logs)


# [CALLBACK FOR TRAIN INTERRUPTION]

###############################################################################
class StopTrainingCallback(Callback):

    # -------------------------------------------------------------------------
    def __init__(self, should_stop: Callable[[], bool] | None = None) -> None:
        super().__init__()
        self.should_stop = should_stop

    # -------------------------------------------------------------------------
    def on_batch_end(self, batch, logs: dict | None = None) -> None:
        if (
            self.should_stop is not None
            and self.should_stop()
            and self.model is not None
        ):
            logger.info("Stop requested; halting training after batch %s", batch)
            self.model.stop_training = True

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        if (
            self.should_stop is not None
            and self.should_stop()
            and self.model is not None
        ):
            logger.info("Stop requested; halting training after epoch %s", epoch + 1)
            self.model.stop_training = True


# [CALLBACK FOR WORKER INTERRUPTIONS]

###############################################################################
class TrainingInterruptCallback(Callback):

    # -------------------------------------------------------------------------
    def __init__(self, worker: Any | None = None) -> None:
        super().__init__()
        self.worker = worker

    # -------------------------------------------------------------------------
    def _check_interrupt(self) -> None:
        if self.worker is None:
            return
        checker = getattr(self.worker, "is_interrupted", None)
        if callable(checker) and checker():
            logger.info("Worker interruption detected; stopping training now.")
            raise WorkerInterrupted()

    # -------------------------------------------------------------------------
    def on_batch_end(self, batch, logs: dict | None = None) -> None:
        self._check_interrupt()

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        self._check_interrupt()


# [CALLBACK FOR PERIODIC CHECKPOINTS]

###############################################################################
class PeriodicCheckpointCallback(Callback):

    # -------------------------------------------------------------------------
    def __init__(self, checkpoint_dir: str, frequency: int = 1) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.frequency = max(1, int(frequency))

    # -------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs: dict | None = None) -> None:
        if (epoch + 1) % self.frequency != 0:
            return
        filename = f"model_checkpoint_E{epoch + 1:02d}.keras"
        target_path = Path(self.checkpoint_dir) / filename
        try:
            if self.model is not None:
                self.model.save(target_path)
                logger.info("Saved checkpoint %s", target_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save checkpoint: %s", exc)

###############################################################################
def build_training_callbacks(
    configuration: dict[str, Any],
    checkpoint_path: str | None,
    total_epochs: int,
    start_epoch: int = 0,
    should_stop: Callable[[], bool] | None = None,
    on_epoch_end: Callable[[int, int, dict[str, Any]], None] | None = None,
    worker: Any | None = None,
) -> list[Callback]:
    progress_messenger = WorkerProgressMessenger(worker=worker)
    callbacks_list: list[Callback] = [
        TrainingInterruptCallback(worker=worker),
        TrainingProgressCallback(
            total_epochs=total_epochs,
            progress_callback=progress_messenger,
            on_epoch_end=on_epoch_end,
            start_epoch=start_epoch,
            polling_interval=configuration.get("polling_interval", 1.0),
        ),
        StopTrainingCallback(should_stop=should_stop),
        keras.callbacks.TerminateOnNaN(),
    ]

    if configuration.get("save_checkpoints", False) and checkpoint_path:
        frequency = int(configuration.get("checkpoints_frequency", 1))
        callbacks_list.append(
            PeriodicCheckpointCallback(
                checkpoint_dir=checkpoint_path,
                frequency=frequency,
            )
        )

    return callbacks_list
