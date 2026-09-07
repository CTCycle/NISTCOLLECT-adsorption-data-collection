from __future__ import annotations

from types import SimpleNamespace

import pytest

from adsmod_ml.learning.training.manager import TrainingManager, TrainingProcessRunner


###############################################################################
def test_resume_validation_accepts_keras3_public_loss_state() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=object())
    TrainingProcessRunner.validate_resume_model(object(), model)


###############################################################################
def test_resume_validation_rejects_missing_loss() -> None:
    optimizer = SimpleNamespace(variables=[object()])
    model = SimpleNamespace(optimizer=optimizer, loss=None)
    with pytest.raises(ValueError, match="not compiled"):
        TrainingProcessRunner.validate_resume_model(object(), model)


###############################################################################
def test_reconstructed_history_uses_one_based_epochs() -> None:
    manager = object.__new__(TrainingManager)

    history = manager.build_history_entries(
        {
            "history": {
                "loss": [0.5],
                "val_loss": [0.6],
                "MaskedR2": [0.25],
                "val_MaskedR2": [0.3],
            }
        }
    )

    assert history == [
        {
            "epoch": 1,
            "loss": 0.5,
            "val_loss": 0.6,
            "masked_r2": 0.25,
            "val_masked_r2": 0.3,
        }
    ]
