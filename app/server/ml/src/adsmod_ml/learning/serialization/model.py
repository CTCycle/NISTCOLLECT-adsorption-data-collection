from __future__ import annotations

from typing import Any

import json
import shutil
from datetime import datetime
from pathlib import Path

from keras.models import load_model

from adsmod_ml.contracts.training import TrainingMetadata
from adsmod_ml.learning.metrics import (
    MaskedMeanSquaredError,
    MaskedRSquared,
)
from adsmod_ml.learning.models.embeddings import MolecularEmbedding
from adsmod_ml.learning.models.encoders import (
    PressureSerierEncoder,
    QDecoder,
    StateEncoder,
)
from adsmod_ml.learning.models.transformers import (
    AddNorm,
    FeedForward,
    TransformerEncoder,
)
from adsmod_ml.learning.training.scheduler import LinearDecayLRScheduler
from adsmod_ml.common.utils.logger import logger
from adsmod_ml.common.utils.security import resolve_checkpoint_path

###############################################################################
class ModelSerializer:

    # -------------------------------------------------------------------------
    def __init__(self, checkpoints_dir: Path, model_name: str = "SCADS") -> None:
        self.model_name = model_name
        self.checkpoints_dir = Path(checkpoints_dir).resolve()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    def resolve_checkpoint_path(self, checkpoint_name: str) -> str:
        return resolve_checkpoint_path(self.checkpoints_dir, checkpoint_name)

    # -------------------------------------------------------------------------
    def create_checkpoint_folder(self) -> str:
        today_datetime = datetime.now().strftime("%Y%m%dT%H%M%S")
        checkpoint_path = self.checkpoints_dir / f"{self.model_name}_{today_datetime}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / "configuration").mkdir(parents=True, exist_ok=True)
        logger.debug("Created checkpoint folder at %s", checkpoint_path)

        return str(checkpoint_path)

    # -------------------------------------------------------------------------
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        checkpoint_path = Path(self.resolve_checkpoint_path(checkpoint_name))
        if not checkpoint_path.exists():
            return False
        try:
            shutil.rmtree(checkpoint_path)
            logger.info("Deleted checkpoint: %s", checkpoint_name)
            return True
        except Exception as e:
            logger.error("Failed to delete checkpoint %s: %s", checkpoint_name, e)
            return False

    # -------------------------------------------------------------------------
    def save_pretrained_model(self, model: Any, path: str) -> None:
        path = Path(path)
        model_files_path = path / "saved_model.keras"
        model.save(model_files_path)
        logger.info("Training session is over. Model %s has been saved", path.name)

    # -------------------------------------------------------------------------
    def save_training_configuration(
        self,
        path: str,
        history: dict,
        configuration: dict[str, Any],
        metadata: TrainingMetadata,
    ) -> None:
        path = Path(path)
        configuration_dir = path / "configuration"
        config_path = configuration_dir / "configuration.json"
        metadata_path = configuration_dir / "metadata.json"
        history_path = configuration_dir / "session_history.json"

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(configuration, f, indent=4, default=str)
        with metadata_path.open("w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=4))
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=4, default=str)

        logger.debug(
            "Model configuration, session history and metadata saved for %s",
            path.name,
        )

    # -------------------------------------------------------------------------
    def scan_checkpoints_folder(self) -> list[str]:
        model_folders: list[str] = []
        if not self.checkpoints_dir.exists():
            return model_folders
        for entry in self.checkpoints_dir.iterdir():
            if entry.is_dir():
                try:
                    self.resolve_checkpoint_path(entry.name)
                except ValueError:
                    continue
                has_keras = any(
                    checkpoint_file.suffix == ".keras" and checkpoint_file.is_file()
                    for checkpoint_file in entry.iterdir()
                )
                if has_keras:
                    model_folders.append(entry.name)

        return sorted(model_folders)

    # -------------------------------------------------------------------------
    def load_training_configuration(
        self, path: str
    ) -> tuple[dict, TrainingMetadata, dict]:
        path = Path(path)
        configuration_dir = path / "configuration"
        config_path = configuration_dir / "configuration.json"
        metadata_path = configuration_dir / "metadata.json"
        history_path = configuration_dir / "session_history.json"
        with config_path.open(encoding="utf-8") as f:
            configuration = json.load(f)
        with metadata_path.open(encoding="utf-8") as f:
            metadata_dict = json.load(f)
            metadata = TrainingMetadata(**metadata_dict)
        with history_path.open(encoding="utf-8") as f:
            history = json.load(f)

        return configuration, metadata, history

    # -------------------------------------------------------------------------
    def load_checkpoint(
        self, checkpoint: str
    ) -> tuple[Any, dict, TrainingMetadata, dict, str]:
        custom_objects = {
            "MaskedMeanSquaredError": MaskedMeanSquaredError,
            "MaskedRSquared": MaskedRSquared,
            "LinearDecayLRScheduler": LinearDecayLRScheduler,
            "MolecularEmbedding": MolecularEmbedding,
            "StateEncoder": StateEncoder,
            "PressureSerierEncoder": PressureSerierEncoder,
            "QDecoder": QDecoder,
            "TransformerEncoder": TransformerEncoder,
            "AddNorm": AddNorm,
            "FeedForward": FeedForward,
        }

        checkpoint_path = self.resolve_checkpoint_path(checkpoint)
        model_path = Path(checkpoint_path) / "saved_model.keras"
        model = load_model(
            model_path,
            custom_objects=custom_objects,
            compile=True,
            # Checkpoints are created and resolved strictly below ADSMOD's configured
            # checkpoint root. Older trusted SCADS Atomic checkpoints may still contain
            # Lambda layers, so keep local checkpoint loading explicit about that trust.
            safe_mode=False,
        )
        configuration, metadata, session = self.load_training_configuration(
            checkpoint_path
        )

        return model, configuration, metadata, session, checkpoint_path
