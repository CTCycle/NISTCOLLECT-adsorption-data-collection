from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from adsmod_common.config import AdsmodConfig
from adsmod_common.training_data import SnapshotPayload, SnapshotReference
from adsmod_core.api import SnapshotDatasetSelection
from adsmod_core.persistence.paths import resolve_storage_root
from adsmod_core.persistence.snapshots import SnapshotStore
from adsmod_core.repositories.database.initializer import prepare_database_for_startup
from adsmod_core.services.container import CoreServiceContainer


###############################################################################
def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


###############################################################################
class TrainingDataService:

    # -------------------------------------------------------------------------
    def __init__(self, container: CoreServiceContainer, *, owns_database: bool = False) -> None:
        self.container = container
        self.snapshot_store = SnapshotStore(container.database)
        self.owns_database = owns_database

    # -------------------------------------------------------------------------
    def close(self) -> None:
        if self.owns_database:
            self.container.database.dispose()

    # -------------------------------------------------------------------------
    def list_sources(self) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        nist_count = int(self.container.nist_repository.count_nist_rows().get("single_component_rows", 0))
        if nist_count > 0:
            sources.append({"source": "nist", "dataset_name": "NIST ISODB", "display_name": "NIST Single Component", "row_count": nist_count, "dataset_id": None})
        sources.extend({"source": item["source"], "dataset_name": item["name"], "display_name": item["name"], "row_count": item["observation_count"], "dataset_id": item["id"]} for item in self.container.datasets.list_summaries() if item["source"] == "uploaded")
        return sources

    # -------------------------------------------------------------------------
    def create_snapshot(self, rows: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> SnapshotReference:
        record = self.snapshot_store.create(rows, metadata=metadata)
        return SnapshotReference(record.snapshot_id, record.content_hash)

    # -------------------------------------------------------------------------
    def create_snapshot_from_selections(self, selections: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> SnapshotReference:
        validated = [SnapshotDatasetSelection.model_validate(selection) for selection in selections]
        rows: list[dict[str, Any]] = []
        for selection in validated:
            if selection.source == "uploaded":
                rows.extend(self._uploaded_snapshot_rows(selection.dataset_name, selection.dataset_id))
            else:
                rows.extend(self._nist_snapshot_rows(selection.dataset_name))
        return self.create_snapshot(rows, metadata={**dict(metadata or {}), "selections": [selection.model_dump(mode="json") for selection in validated]})

    # -------------------------------------------------------------------------
    def fetch_snapshot(self, snapshot_id: str) -> SnapshotPayload:
        rows: list[dict[str, Any]] = []
        page_number = 1
        content_hash: str | None = None
        total_rows: int | None = None
        while total_rows is None or len(rows) < total_rows:
            page = self.snapshot_store.get_page(snapshot_id, page_number, 1000)
            if content_hash is None:
                content_hash = page.content_hash
                total_rows = page.total_rows
            elif page.content_hash != content_hash:
                raise RuntimeError("Snapshot content hash changed during read.")
            rows.extend(dict(row) for row in page.rows)
            page_number += 1
        payload = json.dumps(tuple(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        computed_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if content_hash is None or computed_hash != content_hash:
            raise RuntimeError("Snapshot content hash verification failed.")
        return SnapshotPayload(snapshot_id, content_hash, tuple(rows))

    # -------------------------------------------------------------------------
    def _uploaded_snapshot_rows(self, dataset_name: str, dataset_id: int | None) -> list[dict[str, Any]]:
        summaries = self.container.datasets.list_summaries()
        dataset = next((item for item in summaries if item["source"] == "uploaded" and ((dataset_id is None and item["name"] == dataset_name) or item["id"] == dataset_id)), None)
        if dataset is None:
            raise LookupError(f"Uploaded dataset '{dataset_name}' does not exist.")
        frame = self.container.datasets.observation_frame(int(dataset["id"]))
        if frame.empty:
            raise ValueError(f"Uploaded dataset '{dataset_name}' contains no observations.")
        rows: list[dict[str, Any]] = []
        for index, raw_record in enumerate(frame.to_dict(orient="records")):
            record = {str(key): _json_safe(value) for key, value in raw_record.items()}
            experiment = str(record.get("experiment") or f"row-{index}")
            record.update({"filename": f"{dataset_name}:{experiment}", "temperature": record.get("temperature"), "adsorbent_name": record.get("adsorbent_name"), "adsorbate_name": record.get("adsorbate_name"), "pressure_units": "Pa", "adsorption_units": "mol/kg"})
            rows.append(record)
        return rows

    # -------------------------------------------------------------------------
    def _nist_snapshot_rows(self, dataset_name: str) -> list[dict[str, Any]]:
        adsorption, guests, _ = self.container.nist_repository.load_adsorption_datasets()
        if adsorption.empty:
            raise ValueError(f"NIST dataset '{dataset_name}' contains no observations.")
        guest_properties = {str(row.get("name", "")).strip().casefold(): row for row in guests.to_dict(orient="records")}
        rows: list[dict[str, Any]] = []
        for index, raw_record in enumerate(adsorption.to_dict(orient="records")):
            record = {str(key): _json_safe(value) for key, value in raw_record.items()}
            adsorbate = str(record.get("adsorbate") or "").strip()
            adsorbent = str(record.get("adsorbent") or "").strip()
            guest = guest_properties.get(adsorbate.casefold(), {})
            record.update({"filename": f"{dataset_name}:{record.get('external_key') or index}", "temperature": record.get("temperature_k"), "adsorbent_name": adsorbent, "adsorbate_name": adsorbate, "adsorbate_molecular_weight": _json_safe(guest.get("molecular_weight")), "adsorbate_SMILE": _json_safe(guest.get("smile_code")), "pressure_units": "Pa", "adsorption_units": "mol/kg"})
            rows.append(record)
        return rows


###############################################################################
def open_training_data_service(config: AdsmodConfig) -> TrainingDataService:
    storage_root = resolve_storage_root(config)
    storage_root.mkdir(parents=True, exist_ok=True)
    prepare_database_for_startup(config.application.database, storage_root=storage_root)
    return TrainingDataService(CoreServiceContainer(config), owns_database=True)


__all__ = ["TrainingDataService", "open_training_data_service"]
