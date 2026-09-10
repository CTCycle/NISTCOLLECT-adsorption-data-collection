from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import event

from adsmod_common.config import DatabaseConfig
from adsmod_core.persistence.snapshots import SnapshotStore
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.schemas import Base


###############################################################################
@pytest.fixture
def database() -> Iterator[DatabaseManager]:
    manager = DatabaseManager(
        DatabaseConfig(
            embedded_database=True,
            connect_timeout=5,
            insert_batch_size=100,
            sqlite_path=":memory:",
        )
    )
    Base.metadata.create_all(manager.engine)
    try:
        yield manager
    finally:
        manager.dispose()


###############################################################################
def _rows() -> list[dict[str, int | str]]:
    return [{"id": index, "value": f"value-{index}"} for index in range(7)]


###############################################################################
@pytest.mark.parametrize(
    ("page_number", "expected_ids"),
    [
        pytest.param(1, (0, 1, 2), id="first-page"),
        pytest.param(2, (3, 4, 5), id="middle-page"),
        pytest.param(3, (6,), id="last-page"),
        pytest.param(4, (), id="offset-beyond-end"),
    ],
)
def test_get_page_returns_ordered_slice_and_total(
    database: DatabaseManager,
    page_number: int,
    expected_ids: tuple[int, ...],
) -> None:
    store = SnapshotStore(database)
    rows = _rows()
    record = store.create(rows)

    result = store.get_page(record.snapshot_id, page_number, 3)

    assert result.snapshot_id == record.snapshot_id
    assert result.content_hash == record.content_hash
    assert result.page == page_number
    assert result.page_size == 3
    assert result.total_rows == len(rows)
    assert tuple(row["id"] for row in result.rows) == expected_ids
    assert result.rows == tuple(rows[(page_number - 1) * 3 : page_number * 3])


###############################################################################
def test_get_page_uses_ordered_bounded_row_query(
    database: DatabaseManager,
) -> None:
    store = SnapshotStore(database)
    record = store.create(_rows())
    statements: list[str] = []

    def capture_select(
        conn,
        cursor,
        statement,
        parameters,
        context,
        executemany,  # type: ignore[no-untyped-def]
    ) -> None:
        del conn, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(database.engine, "before_cursor_execute", capture_select)
    try:
        result = store.get_page(record.snapshot_id, 2, 3)
    finally:
        event.remove(database.engine, "before_cursor_execute", capture_select)

    row_queries = [
        statement
        for statement in statements
        if "TRAINING_SNAPSHOT_ROWS" in statement.upper()
    ]
    assert len(statements) == 2
    assert len(row_queries) == 1
    row_query = " ".join(row_queries[0].upper().split())
    assert "WHERE TRAINING_SNAPSHOT_ROWS.SNAPSHOT_ID" in row_query
    assert "ORDER BY TRAINING_SNAPSHOT_ROWS.ROW_INDEX" in row_query
    assert "LIMIT" in row_query
    assert "OFFSET" in row_query
    assert tuple(row["id"] for row in result.rows) == (3, 4, 5)


###############################################################################
def test_get_page_preserves_validation_and_missing_snapshot_errors(
    database: DatabaseManager,
) -> None:
    store = SnapshotStore(database)

    with pytest.raises(ValueError, match="page must be >= 1"):
        store.get_page("missing", 0, 3)
    with pytest.raises(ValueError, match="page_size must be between 1 and 1000"):
        store.get_page("missing", 1, 0)
    with pytest.raises(KeyError):
        store.get_page("missing", 1, 3)
