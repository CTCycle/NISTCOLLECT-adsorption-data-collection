from __future__ import annotations

from typing import Any

import keras

from adsmod_ml.common.constants import PAD_VALUE


# [LOSS FUNCTION]

###############################################################################
@keras.saving.register_keras_serializable(
    package="CustomLoss", name="MaskedMeanSquaredError"
)
class MaskedMeanSquaredError(keras.losses.Loss):

    # -------------------------------------------------------------------------
    def __init__(self, name: str = "MaskedMeanSquaredError", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    # -------------------------------------------------------------------------
    def call(self, y_true: Any, y_pred: Any) -> Any:
        mask = keras.ops.not_equal(y_true, PAD_VALUE)
        mask = keras.ops.cast(mask, dtype=y_true.dtype)
        y_pred = keras.ops.squeeze(y_pred, axis=-1)
        loss = keras.ops.square(y_true - y_pred)
        loss *= mask
        loss = keras.ops.sum(loss) / (keras.ops.sum(mask) + keras.backend.epsilon())

        return loss

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "name": self.name}

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[MaskedMeanSquaredError], config: dict[str, Any]
    ) -> MaskedMeanSquaredError:
        return cls(**config)


# [METRICS]

###############################################################################
@keras.saving.register_keras_serializable(
    package="CustomMetrics", name="MaskedRSquared"
)
class MaskedRSquared(keras.metrics.Metric):

    # -------------------------------------------------------------------------
    def __init__(self, name: str = "MaskedR2", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.ssr = self.add_weight(name="ssr", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    # -------------------------------------------------------------------------
    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        y_pred = keras.ops.squeeze(y_pred, axis=-1)

        y_true = keras.ops.cast(y_true, dtype="float32")
        y_pred = keras.ops.cast(y_pred, dtype="float32")

        mask = keras.ops.not_equal(y_true, PAD_VALUE)
        mask = keras.ops.cast(mask, dtype="float32")

        residuals = keras.ops.square(y_true - y_pred)
        residuals = keras.ops.multiply(residuals, mask)

        mean_y_true = keras.ops.sum(y_true * mask) / (
            keras.ops.sum(mask) + keras.backend.epsilon()
        )
        total_variance = keras.ops.square(y_true - mean_y_true)
        total_variance = keras.ops.multiply(total_variance, mask)

        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, dtype="float32")
            residuals = keras.ops.multiply(residuals, sample_weight)
            total_variance = keras.ops.multiply(total_variance, sample_weight)
            mask = keras.ops.multiply(mask, sample_weight)

        self.ssr.assign_add(keras.ops.sum(residuals))
        self.sst.assign_add(keras.ops.sum(total_variance))
        self.count.assign_add(keras.ops.sum(mask))

    # -------------------------------------------------------------------------
    def result(self) -> Any:
        return 1 - (self.ssr / (self.sst + keras.backend.epsilon()))

    # -------------------------------------------------------------------------
    def reset_states(self) -> None:
        self.ssr.assign(0)
        self.sst.assign(0)
        self.count.assign(0)

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        base_config = super().get_config()
        return {**base_config, "name": self.name}

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[MaskedRSquared], config: dict[str, Any]
    ) -> MaskedRSquared:
        return cls(**config)
