from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.schemas.models import (
    TrainingSnapshot,
    TrainingSnapshotRow,
)


###############################################################################
@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    content_hash: str
    created_at: str
    row_count: int
    rows: tuple[dict[str, Any], ...]


###############################################################################
@dataclass(frozen=True)
class SnapshotPage:
    snapshot_id: str
    content_hash: str
    page: int
    page_size: int
    total_rows: int
    rows: tuple[dict[str, Any], ...]


###############################################################################
class SnapshotStore:
    """Core-owned immutable snapshot storage in the operational database."""

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    @staticmethod
    def _canonical_rows(
        rows: list[dict[str, Any]],
    ) -> tuple[str, tuple[dict[str, Any], ...]]:
        frozen_rows = tuple(dict(row) for row in rows)
        payload = json.dumps(
            frozen_rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return payload, frozen_rows

    # -------------------------------------------------------------------------
    def create(
        self,
        rows: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotRecord:
        if not rows:
            raise ValueError("A snapshot must contain at least one row.")
        payload, frozen_rows = self._canonical_rows(rows)
        content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        snapshot_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        with self.database.transaction() as session:
            snapshot = TrainingSnapshot(
                snapshot_id=snapshot_id,
                content_hash=content_hash,
                created_at=created_at,
                row_count=len(frozen_rows),
                snapshot_metadata=dict(metadata or {}),
                rows=[
                    TrainingSnapshotRow(
                        row_index=index,
                        payload=row,
                    )
                    for index, row in enumerate(frozen_rows)
                ],
            )
            session.add(snapshot)
        return SnapshotRecord(
            snapshot_id=snapshot_id,
            content_hash=content_hash,
            created_at=created_at.isoformat(),
            row_count=len(frozen_rows),
            rows=frozen_rows,
        )

    # -------------------------------------------------------------------------
    def get_page(self, snapshot_id: str, page: int, page_size: int) -> SnapshotPage:
        if page < 1:
            raise ValueError("page must be >= 1")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        with self.database.session_factory() as session:
            snapshot = session.scalar(
                select(TrainingSnapshot)
                .where(TrainingSnapshot.snapshot_id == snapshot_id)
                .options(selectinload(TrainingSnapshot.rows))
            )
            if snapshot is None:
                raise KeyError(snapshot_id)
            rows = tuple(
                dict(row.payload)
                for row in sorted(snapshot.rows, key=lambda row: row.row_index)
            )
            offset = (page - 1) * page_size
            return SnapshotPage(
                snapshot_id=snapshot.snapshot_id,
                content_hash=snapshot.content_hash,
                page=page,
                page_size=page_size,
                total_rows=snapshot.row_count,
                rows=rows[offset : offset + page_size],
            )


__all__ = ["SnapshotPage", "SnapshotRecord", "SnapshotStore"]
