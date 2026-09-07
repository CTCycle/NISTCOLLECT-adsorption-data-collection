from __future__ import annotations

from sqlalchemy import UniqueConstraint
from sqlalchemy.sql.schema import Table

###############################################################################
class ConflictCandidateRanker:

    # -------------------------------------------------------------------------
    def __init__(self, table: Table, column_positions: dict[str, int]) -> None:
        self.table = table
        self.column_positions = column_positions

    # -------------------------------------------------------------------------
    def __call__(
        self, columns: list[str]
    ) -> tuple[int, int, int, tuple[int, ...], tuple[str, ...]]:
        lowered = tuple(column.lower() for column in columns)
        all_not_nullable = all(not self.table.c[column].nullable for column in columns)
        contains_key_marker = any(
            column.endswith("_key") or column == "content_hash" for column in lowered
        )
        ordered_positions = tuple(
            self.column_positions.get(column, 10000) for column in columns
        )
        return (
            0 if all_not_nullable else 1,
            0 if contains_key_marker else 1,
            len(columns),
            ordered_positions,
            lowered,
        )

###############################################################################
def resolve_conflict_columns(table: Table) -> list[str]:
    unique_constraints = [
        [column.name for column in constraint.columns]
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    ]
    if not unique_constraints:
        primary_key_columns = [column.name for column in table.primary_key.columns]
        if primary_key_columns:
            return primary_key_columns
        raise ValueError(f"No unique or primary key constraint found for {table.name}")

    column_positions = {
        column.name: index for index, column in enumerate(table.columns)
    }
    ranker = ConflictCandidateRanker(table, column_positions)
    return list(min(unique_constraints, key=ranker))
