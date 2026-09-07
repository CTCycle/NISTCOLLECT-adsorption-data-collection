from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from adsmod_common.training_data import TrainingDataAccess
from adsmod_ml.contracts.training import TrainingMetadata


###############################################################################
class TrainingDataSerializer:
    """ML-side manifest for backend-owned immutable training snapshots."""

    dataset_label_column = "dataset_label"
    dataset_hash_column = "dataset_hash"
    dataset_name_column = "dataset_name"
    sample_key_column = "sample_key"
    series_columns = ["pressure", "adsorbed_amount", "adsorbate_encoded_SMILE"]

    # -------------------------------------------------------------------------
    def __init__(
        self, snapshot_access: TrainingDataAccess, artifact_root: Path
    ) -> None:
        self.snapshot_access = snapshot_access
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.artifact_root / "training-manifest.json"

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_dataset_label(dataset_label: str | None) -> str:
        normalized = str(dataset_label or "").strip()
        if not normalized:
            raise ValueError("dataset_label is required.")
        return normalized

    # -------------------------------------------------------------------------
    @classmethod
    def build_sample_key(cls, row: pd.Series) -> str:
        payload = {
            cls.dataset_label_column: row.get(cls.dataset_label_column),
            cls.dataset_name_column: row.get(cls.dataset_name_column),
            "split": row.get("split"),
            "temperature": row.get("temperature"),
            "pressure": row.get("pressure"),
            "adsorbed_amount": row.get("adsorbed_amount"),
            "encoded_adsorbent": row.get("encoded_adsorbent"),
            "adsorbate_molecular_weight": row.get("adsorbate_molecular_weight"),
            "adsorbate_encoded_SMILE": row.get("adsorbate_encoded_SMILE"),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_sequence_value(value: Any) -> list[Any]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                return []
            try:
                parsed = json.loads(trimmed)
            except json.JSONDecodeError:
                return [item.strip() for item in trimmed.split(",") if item.strip()]
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return list(parsed.values())
            return [parsed]
        if isinstance(value, pd.Series):
            return value.tolist()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return list(tolist())
        return [value]

    # -------------------------------------------------------------------------
    def coerce_sequence_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if dataset.empty:
            return dataset.copy()
        normalized = dataset.copy()
        for column in self.series_columns:
            if column in normalized.columns:
                normalized[column] = normalized[column].apply(self.parse_sequence_value)
        return normalized

    # -------------------------------------------------------------------------
    @staticmethod
    def _jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int, float)):
            if isinstance(value, float) and pd.isna(value):
                return None
            return value
        if isinstance(value, dict):
            return {
                str(key): TrainingDataSerializer._jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [TrainingDataSerializer._jsonable(item) for item in value]
        item = getattr(value, "item", None)
        if callable(item):
            return TrainingDataSerializer._jsonable(item())
        if pd.isna(value):
            return None
        return str(value)

    # -------------------------------------------------------------------------
    def _read_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.is_file():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("The ML training manifest is unreadable.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("The ML training manifest must be a JSON object.")
        return {
            str(label): dict(entry)
            for label, entry in payload.items()
            if isinstance(entry, dict)
        }

    # -------------------------------------------------------------------------
    def _write_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    # -------------------------------------------------------------------------
    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: set[str], context: str) -> None:
        missing = columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{context} is missing required columns: {', '.join(sorted(missing))}"
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def require_dataset_hash(dataset_hash: Any) -> str:
        normalized = (
            ""
            if dataset_hash is None or pd.isna(dataset_hash)
            else str(dataset_hash).strip()
        )
        if len(normalized) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in normalized
        ):
            raise ValueError("dataset_hash must be a 64-character hexadecimal digest.")
        return normalized

    # -------------------------------------------------------------------------
    def save_training_dataset(
        self,
        dataset: pd.DataFrame,
        dataset_label: str,
        dataset_hash: str,
    ) -> None:
        if dataset.empty:
            raise ValueError("Training dataset must not be empty.")
        label = self.normalize_dataset_label(dataset_label)
        content_hash = self.require_dataset_hash(dataset_hash)
        normalized = self.coerce_sequence_columns(dataset)
        if self.dataset_label_column not in normalized.columns:
            normalized[self.dataset_label_column] = label
        normalized[self.dataset_hash_column] = content_hash
        if self.sample_key_column not in normalized.columns:
            normalized[self.sample_key_column] = normalized.apply(
                self.build_sample_key, axis=1
            )
        normalized = normalized.drop_duplicates(
            subset=[self.sample_key_column], keep="last"
        )
        rows = [
            {str(key): self._jsonable(value) for key, value in record.items()}
            for record in normalized.to_dict(orient="records")
        ]
        reference = self.snapshot_access.create_snapshot(
            rows,
            metadata={
                "kind": "processed_training",
                "dataset_label": label,
                "dataset_hash": content_hash,
            },
        )
        manifest = self._read_manifest()
        previous = manifest.get(label, {})
        manifest[label] = {
            "snapshot_id": reference.snapshot_id,
            "snapshot_hash": reference.content_hash,
            "dataset_hash": content_hash,
            "created_at": previous.get("created_at")
            or datetime.now(timezone.utc).isoformat(),
            "metadata": previous.get("metadata", {}),
        }
        self._write_manifest(manifest)

    # -------------------------------------------------------------------------
    def save_training_metadata(
        self, metadata: pd.DataFrame, dataset_label: str
    ) -> None:
        if metadata.empty:
            return
        label = self.normalize_dataset_label(dataset_label)
        row = metadata.iloc[0].to_dict()
        row.pop(self.dataset_label_column, None)
        row["dataset_hash"] = self.require_dataset_hash(row.get("dataset_hash"))
        for key in ("smile_vocabulary", "adsorbent_vocabulary", "normalization_stats"):
            value = row.get(key)
            if isinstance(value, str):
                try:
                    row[key] = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Training metadata field '{key}' is invalid JSON."
                    ) from exc
        manifest = self._read_manifest()
        entry = manifest.get(label)
        if entry is None:
            raise ValueError("Training dataset must be saved before its metadata.")
        entry["metadata"] = self._jsonable(row)
        entry["dataset_hash"] = row["dataset_hash"]
        manifest[label] = entry
        self._write_manifest(manifest)

    # -------------------------------------------------------------------------
    def load_training_metadata(self, dataset_label: str) -> TrainingMetadata:
        label = self.normalize_dataset_label(dataset_label)
        entry = self._read_manifest().get(label)
        if not entry or not isinstance(entry.get("metadata"), dict):
            return TrainingMetadata()
        return TrainingMetadata.model_validate(entry["metadata"])

    # -------------------------------------------------------------------------
    def load_training_data(
        self,
        dataset_label: str,
        only_metadata: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, TrainingMetadata] | TrainingMetadata:
        label = self.normalize_dataset_label(dataset_label)
        metadata = self.load_training_metadata(label)
        if only_metadata:
            return metadata
        entry = self._read_manifest().get(label)
        if not entry or not entry.get("snapshot_id"):
            empty = pd.DataFrame()
            return empty, empty, metadata
        payload = self.snapshot_access.fetch_snapshot(str(entry["snapshot_id"]))
        if payload.content_hash != entry.get("snapshot_hash"):
            raise RuntimeError("Snapshot hash does not match the ML manifest.")
        data = self.coerce_sequence_columns(pd.DataFrame(list(payload.rows)))
        self._require_columns(
            data,
            {self.dataset_label_column, self.dataset_hash_column, "split"},
            "Training data",
        )
        data = data[data[self.dataset_label_column] == label].copy()
        return (
            data[data["split"] == "train"].reset_index(drop=True),
            data[data["split"] == "validation"].reset_index(drop=True),
            metadata,
        )

    # -------------------------------------------------------------------------
    def collect_dataset_hashes(self) -> set[str]:
        return {
            str(entry["dataset_hash"])
            for entry in self._read_manifest().values()
            if entry.get("dataset_hash")
        }

    # -------------------------------------------------------------------------
    def clear_training_dataset(self, dataset_label: str | None = None) -> None:
        manifest = self._read_manifest()
        if dataset_label is None:
            manifest.clear()
        else:
            manifest.pop(self.normalize_dataset_label(dataset_label), None)
        self._write_manifest(manifest)

    # -------------------------------------------------------------------------
    def list_processed_datasets(self) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        for label, entry in sorted(self._read_manifest().items()):
            metadata = (
                entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            )
            datasets.append(
                {
                    "dataset_label": label,
                    "dataset_hash": entry.get("dataset_hash"),
                    "train_samples": int(metadata.get("train_samples", 0)),
                    "validation_samples": int(metadata.get("validation_samples", 0)),
                    "created_at": entry.get("created_at"),
                }
            )
        return datasets

    # -------------------------------------------------------------------------
    def get_training_dataset_info(self, dataset_label: str) -> dict[str, Any] | None:
        label = self.normalize_dataset_label(dataset_label)
        metadata = self.load_training_metadata(label)
        if not metadata.dataset_hash or metadata.total_samples <= 0:
            return None
        return {
            "dataset_label": label,
            **metadata.model_dump(),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_metadata_hash(metadata: TrainingMetadata) -> str:
        payload = {
            "sample_size": metadata.sample_size,
            "validation_size": metadata.validation_size,
            "min_measurements": metadata.min_measurements,
            "max_measurements": metadata.max_measurements,
            "smile_sequence_size": metadata.smile_sequence_size,
            "max_pressure": metadata.max_pressure,
            "max_uptake": metadata.max_uptake,
            "smile_vocabulary": sorted(metadata.smile_vocabulary.items()),
            "adsorbent_vocabulary": sorted(metadata.adsorbent_vocabulary.items()),
            "normalization_stats": metadata.normalization_stats,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def validate_metadata(
        metadata: TrainingMetadata, target_metadata: TrainingMetadata
    ) -> bool:
        if not metadata or not target_metadata:
            return False
        return TrainingDataSerializer.compute_metadata_hash(
            metadata
        ) == TrainingDataSerializer.compute_metadata_hash(target_metadata)


__all__ = ["TrainingDataSerializer"]
