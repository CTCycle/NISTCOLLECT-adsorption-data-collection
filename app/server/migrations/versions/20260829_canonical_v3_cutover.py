"""Move the operational schema to the canonical Core-owned v3 model.

Revision ID: 20260829_v3
Revises: 23f1110c64a9
"""

from __future__ import annotations

from typing import Sequence

from alembic import op
import sqlalchemy as sa

from adsmod_core.repositories.schemas import types as schema_types


revision: str = "20260829_v3"
down_revision: str | None = "23f1110c64a9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


###############################################################################
def upgrade() -> None:
    # Training data is now represented by immutable Core snapshots. Existing
    # processed training tables are intentionally retired; callers must build
    # a new snapshot through the Core API.
    op.drop_table("training_samples")
    op.drop_table("training_datasets")

    with op.batch_alter_table("fitting_runs") as batch_op:
        batch_op.drop_column("best_result_id")
        batch_op.drop_column("dataset_id")
        batch_op.drop_column("component_id")

    op.create_table(
        "training_snapshots",
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", schema_types.UTCDateTime(timezone=True), nullable=False
        ),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("metadata", schema_types.JSONMapping(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("content_hash", name="uq_training_snapshots_content_hash"),
        sa.CheckConstraint("row_count > 0", name="ck_training_snapshots_row_count"),
    )
    op.create_table(
        "training_snapshot_rows",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_id", sa.String(length=36), nullable=False),
        sa.Column("row_index", sa.Integer(), nullable=False),
        sa.Column("payload", schema_types.JSONMapping(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["training_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "snapshot_id", "row_index", name="uq_training_snapshot_rows_index"
        ),
        sa.CheckConstraint("row_index >= 0", name="ck_training_snapshot_rows_index"),
    )
    with op.batch_alter_table("training_snapshot_rows") as batch_op:
        batch_op.create_index(
            "ix_training_snapshot_rows_snapshot_index",
            ["snapshot_id", "row_index"],
            unique=False,
        )


###############################################################################
def downgrade() -> None:
    raise RuntimeError(
        "The v3 cutover is intentionally irreversible: recreate a pre-v3 database "
        "from an export instead of restoring retired training persistence."
    )
