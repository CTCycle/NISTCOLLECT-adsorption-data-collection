from __future__ import annotations

from typing import Any

from keras import Model
from keras.utils import set_random_seed

from adsmod_ml.learning.callbacks import build_training_callbacks
from adsmod_ml.learning.device import DeviceConfig, DeviceDataLoader


# [TOOLS FOR TRAINING MACHINE LEARNING MODELS]

###############################################################################
class ModelTraining:

    # -------------------------------------------------------------------------
    def __init__(
        self, configuration: dict[str, Any], metadata: dict[str, Any] | None = None
    ) -> None:
        set_random_seed(configuration.get("training_seed", 42))
        self.configuration = configuration
        self.metadata = metadata

    # -------------------------------------------------------------------------
    @staticmethod
    def _to_generator(loader: Any) -> Any:
        while True:
            for batch in loader:
                yield batch

    # -------------------------------------------------------------------------
    def train_model(
        self,
        model: Model,
        train_data: Any,
        validation_data: Any,
        checkpoint_path: str | None,
        **kwargs,
    ) -> tuple[Model, dict[str, Any]]:
        total_epochs = int(self.configuration.get("epochs", 10))
        callbacks_list = build_training_callbacks(
            self.configuration,
            checkpoint_path,
            total_epochs=total_epochs,
            start_epoch=0,
            should_stop=kwargs.get("should_stop"),
            on_epoch_end=kwargs.get("on_epoch_end"),
            worker=kwargs.get("worker"),
        )

        device = DeviceConfig(self.configuration).set_device()
        train_data = DeviceDataLoader(train_data, device)
        validation_data = DeviceDataLoader(validation_data, device)

        session = model.fit(
            self._to_generator(train_data),
            steps_per_epoch=len(train_data),
            validation_data=self._to_generator(validation_data),
            validation_steps=len(validation_data),
            epochs=total_epochs,
            callbacks=callbacks_list,
        )

        history = {"history": session.history, "epochs": session.epoch[-1] + 1}

        return model, history

    # -------------------------------------------------------------------------
    def resume_training(
        self,
        model: Model,
        train_data: Any,
        validation_data: Any,
        checkpoint_path: str | None,
        session: dict | None = None,
        additional_epochs: int = 10,
        **kwargs,
    ) -> tuple[Model, dict[str, Any]]:
        session = session or {}
        from_epoch = session.get("epochs", 0)
        total_epochs = from_epoch + additional_epochs
        callbacks_list = build_training_callbacks(
            self.configuration,
            checkpoint_path,
            total_epochs=total_epochs,
            start_epoch=from_epoch,
            should_stop=kwargs.get("should_stop"),
            on_epoch_end=kwargs.get("on_epoch_end"),
            worker=kwargs.get("worker"),
        )

        device = DeviceConfig(self.configuration).set_device()
        train_data = DeviceDataLoader(train_data, device)
        validation_data = DeviceDataLoader(validation_data, device)

        new_session = model.fit(
            self._to_generator(train_data),
            steps_per_epoch=len(train_data),
            validation_data=self._to_generator(validation_data),
            validation_steps=len(validation_data),
            epochs=total_epochs,
            callbacks=callbacks_list,
            initial_epoch=from_epoch,
        )

        history = {"history": new_session.history, "epochs": new_session.epoch[-1] + 1}
        if session and "history" in session:
            merged_history: dict[str, list[Any]] = {}
            session_history = session.get("history", {})
            if not isinstance(session_history, dict):
                session_history = {}
            history_keys = set(session_history) | set(new_session.history)
            for key in history_keys:
                merged_history[key] = list(session_history.get(key, [])) + list(
                    new_session.history.get(key, [])
                )
            history["history"] = merged_history

        return model, history
