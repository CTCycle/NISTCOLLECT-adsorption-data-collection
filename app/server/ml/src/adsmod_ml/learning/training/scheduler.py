from __future__ import annotations

from typing import Any

import keras
import numpy as np


# [LEARNING RATE SCHEDULER]

###############################################################################
@keras.saving.register_keras_serializable(package="LinearDecayLRScheduler")
class LinearDecayLRScheduler(keras.optimizers.schedules.LearningRateSchedule):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        initial_lr: float,
        constant_steps: int,
        decay_steps: int,
        target_lr: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.initial_lr = initial_lr
        self.constant_steps = constant_steps
        self.decay_steps = decay_steps
        self.target_lr = target_lr

    # -------------------------------------------------------------------------
    def __call__(self, step) -> Any:
        current_step = keras.ops.cast(step, np.float32)
        constant_steps = keras.ops.cast(self.constant_steps, np.float32)
        decay_steps = keras.ops.cast(self.decay_steps, np.float32)
        initial_lr = keras.ops.cast(self.initial_lr, np.float32)
        target_lr = keras.ops.cast(self.target_lr, np.float32)

        progress = (current_step - constant_steps) / decay_steps
        decayed_lr = initial_lr - (initial_lr - target_lr) * progress
        decayed_lr = keras.ops.maximum(decayed_lr, target_lr)

        learning_rate = keras.ops.where(
            current_step < constant_steps, initial_lr, decayed_lr
        )

        return learning_rate

    # -------------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        return {
            "initial_lr": self.initial_lr,
            "constant_steps": self.constant_steps,
            "decay_steps": self.decay_steps,
            "target_lr": self.target_lr,
        }

    # -------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls: type[LinearDecayLRScheduler], config: dict[str, Any]
    ) -> LinearDecayLRScheduler:
        return cls(**config)
