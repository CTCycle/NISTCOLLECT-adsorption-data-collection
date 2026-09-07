from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


###############################################################################
@dataclass(frozen=True)
class SnapshotReference:
    snapshot_id: str
    content_hash: str


###############################################################################
@dataclass(frozen=True)
class SnapshotPayload:
    snapshot_id: str
    content_hash: str
    rows: tuple[dict[str, Any], ...]


###############################################################################
class TrainingDataAccess(Protocol):

    # -------------------------------------------------------------------------
    def list_sources(self) -> list[dict[str, Any]]: ...

    # -------------------------------------------------------------------------
    def create_snapshot(
        self,
        rows: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference: ...

    # -------------------------------------------------------------------------
    def create_snapshot_from_selections(
        self,
        selections: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotReference: ...

    # -------------------------------------------------------------------------
    def fetch_snapshot(self, snapshot_id: str) -> SnapshotPayload: ...


__all__ = ["SnapshotPayload", "SnapshotReference", "TrainingDataAccess"]
