from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

###############################################################################
def _deduplicate(
    records: Iterable[dict[str, Any]], conflict_columns: Sequence[str]
) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for record in records:
        key = tuple(record.get(column) for column in conflict_columns)
        if any(value is None for value in key):
            passthrough.append(record)
        else:
            by_key[key] = record
    return [*by_key.values(), *passthrough]

###############################################################################
def upsert_records(
    session: Session,
    table: Table,
    records: Iterable[dict[str, Any]],
    conflict_columns: Sequence[str],
) -> int:
    """Upsert records using the caller-supplied conflict contract in one session transaction."""
    normalized_records = []
    for record in records:
        normalized = dict(record)
        normalized_records.append(normalized)
    batch = _deduplicate(normalized_records, conflict_columns)
    if not batch:
        return 0
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "sqlite":
        statement = sqlite_insert(table)
    elif dialect_name == "postgresql":
        statement = postgres_insert(table)
    else:
        raise ValueError(f"Unsupported upsert dialect: {dialect_name}")
    excluded = statement.excluded
    conflict_set = set(conflict_columns)
    update_columns = {
        column: excluded[column]
        for column in batch[0]
        if column not in conflict_set and column in table.c
    }
    if update_columns:
        statement = statement.on_conflict_do_update(
            index_elements=list(conflict_columns), set_=update_columns
        )
    else:
        statement = statement.on_conflict_do_nothing(
            index_elements=list(conflict_columns)
        )
    session.execute(statement, batch)
    return len(batch)


__all__ = ["upsert_records"]
