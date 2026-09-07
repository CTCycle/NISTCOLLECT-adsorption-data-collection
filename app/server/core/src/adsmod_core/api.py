from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

###############################################################################
class SnapshotCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


###############################################################################
class SnapshotDatasetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["nist", "uploaded"]
    dataset_name: str = Field(min_length=1, max_length=128)
    dataset_id: int | None = Field(default=None, ge=1)


###############################################################################
class SnapshotFromSelectionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selections: list[SnapshotDatasetSelection] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

###############################################################################
class SnapshotCreateResponse(BaseModel):
    snapshot_id: str
    content_hash: str
    created_at: str
    row_count: int

###############################################################################
class SnapshotPageResponse(BaseModel):
    snapshot_id: str
    content_hash: str
    page: int
    page_size: int
    total_rows: int
    rows: list[dict[str, Any]]
