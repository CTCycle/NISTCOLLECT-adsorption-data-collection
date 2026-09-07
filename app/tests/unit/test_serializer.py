from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from adsmod_common.training_data import SnapshotPayload, SnapshotReference
from adsmod_ml.contracts.training import TrainingMetadata
from adsmod_ml.learning.serialization.model import ModelSerializer
from adsmod_ml.learning.serialization.training import TrainingDataSerializer


###############################################################################
class FakeSnapshotAccess:

    # -------------------------------------------------------------------------
    def __init__(self, response_rows: list[dict[str, Any]] | None = None) -> None:
        self.response_rows = response_rows
        self.captured: list[dict[str, Any]] = []
        self.snapshots: dict[str, SnapshotPayload] = {}

    # -------------------------------------------------------------------------
    @staticmethod
    def _hash(rows: list[dict[str, Any]]) -> str:
        payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    def list_sources(self) -> list[dict[str, Any]]:
        return []

    # -------------------------------------------------------------------------
    def create_snapshot(
        self,
        rows: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference:
        frozen_rows = [dict(row) for row in rows]
        self.captured.append({"rows": frozen_rows, "metadata": dict(metadata or {})})
        content_hash = self._hash(frozen_rows)
        reference = SnapshotReference("snapshot-1", content_hash)
        stored_rows = self.response_rows if self.response_rows is not None else frozen_rows
        self.snapshots[reference.snapshot_id] = SnapshotPayload(
            snapshot_id=reference.snapshot_id,
            content_hash=self._hash(stored_rows),
            rows=tuple(dict(row) for row in stored_rows),
        )
        return reference

    # -------------------------------------------------------------------------
    def create_snapshot_from_selections(
        self,
        selections: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference:
        raise AssertionError("selection snapshots are not used by serializer tests")

    # -------------------------------------------------------------------------
    def fetch_snapshot(self, snapshot_id: str) -> SnapshotPayload:
        return self.snapshots[snapshot_id]


###############################################################################
def create_basis_metadata(**kwargs: object) -> TrainingMetadata:
    defaults: dict[str, object] = {
        "sample_size": 1.0,
        "validation_size": 0.2,
        "min_measurements": 1,
        "max_measurements": 30,
        "smile_sequence_size": 20,
        "max_pressure": 10000.0,
        "max_uptake": 20.0,
        "smile_vocabulary": {"C": 1, "H": 2, "O": 3},
        "adsorbent_vocabulary": {"MOF-1": 1, "ZEOLITE-X": 2},
        "normalization_stats": {"pressure_mean": 5000.0, "pressure_std": 2000.0},
    }
    defaults.update(kwargs)
    return TrainingMetadata(**defaults)


###############################################################################
def _serializer(
    tmp_path: Path,
    *,
    response_rows: list[dict[str, Any]] | None = None,
) -> tuple[TrainingDataSerializer, FakeSnapshotAccess]:
    access = FakeSnapshotAccess(response_rows)
    return TrainingDataSerializer(access, tmp_path / "artifacts"), access


###############################################################################
def test_validate_metadata_identical() -> None:
    assert TrainingDataSerializer.validate_metadata(create_basis_metadata(), create_basis_metadata()) is True


###############################################################################
def test_validate_metadata_rejects_parameter_and_vocabulary_changes() -> None:
    assert TrainingDataSerializer.validate_metadata(
        create_basis_metadata(sample_size=1.0),
        create_basis_metadata(sample_size=0.5),
    ) is False
    assert TrainingDataSerializer.validate_metadata(
        create_basis_metadata(smile_vocabulary={"A": 1}),
        create_basis_metadata(smile_vocabulary={"A": 1, "B": 2}),
    ) is False
    assert TrainingDataSerializer.validate_metadata(
        create_basis_metadata(smile_vocabulary={"A": 1, "B": 2}),
        create_basis_metadata(smile_vocabulary={"A": 2, "B": 1}),
    ) is False


###############################################################################
def test_compute_metadata_hash_is_deterministic() -> None:
    first = create_basis_metadata(smile_vocabulary={"A": 1, "B": 2})
    second = create_basis_metadata(smile_vocabulary={"B": 2, "A": 1})
    assert TrainingDataSerializer.compute_metadata_hash(first) == TrainingDataSerializer.compute_metadata_hash(second)


###############################################################################
def test_save_training_dataset_deduplicates_rows_and_publishes_snapshot(tmp_path: Path) -> None:
    serializer, access = _serializer(tmp_path)
    dataset = pd.DataFrame([
        {
            "dataset_name": "NIST ISODB",
            "split": "train",
            "temperature": 298.15,
            "pressure": "[1.0,2.0]",
            "adsorbed_amount": "[0.1,0.2]",
            "encoded_adsorbent": 1,
            "adsorbate_molecular_weight": 44.0,
            "adsorbate_encoded_SMILE": "[1,2,3]",
        },
        {
            "dataset_name": "NIST ISODB",
            "split": "train",
            "temperature": 298.15,
            "pressure": "[1.0,2.0]",
            "adsorbed_amount": "[0.1,0.2]",
            "encoded_adsorbent": 1,
            "adsorbate_molecular_weight": 44.0,
            "adsorbate_encoded_SMILE": "[1,2,3]",
        },
    ])
    serializer.save_training_dataset(dataset, "small_dataset", "a" * 64)
    rows = access.captured[0]["rows"]
    assert len(rows) == 1
    assert rows[0]["pressure"] == [1.0, 2.0]
    assert rows[0]["adsorbed_amount"] == [0.1, 0.2]
    assert rows[0]["adsorbate_encoded_SMILE"] == [1, 2, 3]


###############################################################################
def test_save_training_metadata_normalizes_json_mappings(tmp_path: Path) -> None:
    serializer, access = _serializer(tmp_path)
    serializer.save_training_dataset(pd.DataFrame([{"split": "train"}]), "small_dataset", "a" * 64)
    metadata = pd.DataFrame([{
        "dataset_label": "small_dataset",
        "dataset_hash": "a" * 64,
        "smile_vocabulary": '{"C": 1}',
        "adsorbent_vocabulary": '{"MOF-1": 0}',
        "normalization_stats": '{"pressure_mean": 1.0}',
    }])
    serializer.save_training_metadata(metadata, "small_dataset")
    manifest = json.loads((tmp_path / "artifacts" / "training-manifest.json").read_text(encoding="utf-8"))
    saved = manifest["small_dataset"]["metadata"]
    assert "dataset_label" not in saved
    assert saved["smile_vocabulary"] == {"C": 1}
    assert saved["adsorbent_vocabulary"] == {"MOF-1": 0}
    assert saved["normalization_stats"] == {"pressure_mean": 1.0}
    loaded = serializer.load_training_metadata("small_dataset")
    assert loaded.dataset_hash == "a" * 64
    assert loaded.smile_vocabulary == {"C": 1}
    assert len(access.captured) == 1


###############################################################################
def test_training_dataset_requires_canonical_hash(tmp_path: Path) -> None:
    serializer, _ = _serializer(tmp_path)
    with pytest.raises(ValueError, match="dataset_hash"):
        serializer.save_training_dataset(pd.DataFrame({"split": ["train"]}), "small_dataset", "")


###############################################################################
def test_training_metadata_rejects_legacy_fields() -> None:
    with pytest.raises(ValidationError):
        TrainingMetadata(hashcode="a" * 64)


###############################################################################
def test_training_data_read_requires_canonical_columns(tmp_path: Path) -> None:
    rows = [{"split": "train"}]
    serializer, access = _serializer(tmp_path, response_rows=rows)
    reference = access.create_snapshot(rows)
    serializer._write_manifest({
        "default": {
            "snapshot_id": reference.snapshot_id,
            "snapshot_hash": reference.content_hash,
            "dataset_hash": "a" * 64,
        }
    })
    with pytest.raises(ValueError, match="dataset_label"):
        serializer.load_training_data("default")


###############################################################################
def test_checkpoint_metadata_rejects_legacy_hash_alias(tmp_path: Path) -> None:
    configuration_dir = tmp_path / "configuration"
    configuration_dir.mkdir()
    (configuration_dir / "configuration.json").write_text("{}", encoding="utf-8")
    (configuration_dir / "metadata.json").write_text(
        '{"hash_code": "' + ("a" * 64) + '"}', encoding="utf-8"
    )
    (configuration_dir / "session_history.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValidationError):
        ModelSerializer(tmp_path).load_training_configuration(str(tmp_path))


###############################################################################
def test_checkpoint_loader_allows_adsmod_owned_lambda_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "local-checkpoint"
    configuration_dir = checkpoint / "configuration"
    configuration_dir.mkdir(parents=True)
    (checkpoint / "saved_model.keras").touch()
    (configuration_dir / "configuration.json").write_text("{}", encoding="utf-8")
    (configuration_dir / "metadata.json").write_text(
        create_basis_metadata().model_dump_json(), encoding="utf-8"
    )
    (configuration_dir / "session_history.json").write_text("{}", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_load_model(path: Path, **kwargs: Any) -> object:
        captured["path"] = path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "adsmod_ml.learning.serialization.model.load_model", fake_load_model
    )
    loaded, _, _, _, resolved = ModelSerializer(tmp_path).load_checkpoint(
        "local-checkpoint"
    )
    assert loaded is not None
    assert captured["safe_mode"] is False
    assert captured["compile"] is True
    assert Path(resolved) == checkpoint.resolve()
