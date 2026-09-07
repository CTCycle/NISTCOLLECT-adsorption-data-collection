from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import sqlalchemy
from sqlalchemy import event, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import IntegrityError, InterfaceError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from adsmod_core.repositories.database.bulk import upsert_records
from adsmod_core.repositories.database.upsert import resolve_conflict_columns
from adsmod_core.repositories.schemas.models import Adsorbate, Base

###############################################################################
def configure_sqlite_connection(dbapi_connection, connection_record) -> None:  # type: ignore[no-untyped-def]
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
    finally:
        cursor.close()

###############################################################################
def run_concurrent_upsert(
    barrier: threading.Barrier,
    engine: sqlalchemy.Engine,
    payload: dict[str, object],
    conflict_columns: list[str],
) -> None:
    barrier.wait()
    run_upsert_with_retry(engine, payload, conflict_columns)

###############################################################################
def run_upsert_with_retry(
    engine: sqlalchemy.Engine,
    payload: dict[str, object],
    conflict_columns: list[str],
    retries: int = 2,
) -> None:
    attempts = 0
    while True:
        try:
            upsert_adsorbate(engine, payload, conflict_columns)
            return
        except InterfaceError:
            attempts += 1
            if attempts > retries:
                raise

###############################################################################
def build_sqlite_engine() -> sqlalchemy.Engine:
    engine = sqlalchemy.create_engine(
        "sqlite://",
        future=True,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
    )
    event.listen(engine, "connect", configure_sqlite_connection)
    Base.metadata.create_all(engine)
    return engine

###############################################################################
def upsert_adsorbate(
    engine: sqlalchemy.Engine,
    payload: dict[str, object],
    conflict_columns: list[str],
) -> None:
    with Session(engine) as session:
        upsert_records(session, Adsorbate.__table__, [payload], conflict_columns)
        session.commit()

###############################################################################
def test_wrong_conflict_target_can_raise_duplicate_adsorbate_identity() -> None:
    engine = build_sqlite_engine()

    first = {
        "key": "name:0001",
        "name": "methane",
        "normalized_name": "methane",
        "inchi_key": None,
        "formula": "CH4",
    }
    second = {
        "key": "name:0001",
        "name": "methane",
        "normalized_name": "methane",
        "inchi_key": None,
        "formula": "CH4-updated",
    }

    upsert_adsorbate(engine, first, ["inchi_key"])
    with Session(engine) as session:
        statement = (
            insert(Adsorbate.__table__)
            .values([second])
            .on_conflict_do_update(
                index_elements=["inchi_key"],
                set_={"formula": "CH4-updated"},
            )
        )
        with pytest.raises(IntegrityError):
            session.execute(statement)
            session.commit()

    engine.dispose()

###############################################################################
def test_retry_upsert_uses_single_adsorbate_row_id() -> None:
    engine = build_sqlite_engine()
    conflict_columns = resolve_conflict_columns(Adsorbate.__table__)
    assert conflict_columns == ["key"]

    first = {
        "key": "name:0002",
        "name": "ethane",
        "normalized_name": "ethane",
        "inchi_key": None,
        "formula": "C2H6",
    }
    second = {
        "key": "name:0002",
        "name": "ethane",
        "normalized_name": "ethane",
        "inchi_key": None,
        "formula": "C2H6-updated",
        "molar_mass_g_mol": 30.07,
    }

    upsert_adsorbate(engine, first, conflict_columns)
    upsert_adsorbate(engine, second, conflict_columns)

    with Session(engine) as session:
        rows = (
            session.execute(select(Adsorbate).where(Adsorbate.key == "name:0002"))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].id is not None
        assert rows[0].formula == "C2H6-updated"
        assert rows[0].molar_mass_g_mol == 30.07

    engine.dispose()

###############################################################################
def test_concurrent_upsert_keeps_single_adsorbate_identity() -> None:
    engine = build_sqlite_engine()
    conflict_columns = resolve_conflict_columns(Adsorbate.__table__)
    barrier = threading.Barrier(2)

    payload_a = {
        "key": "name:0003",
        "name": "propane",
        "normalized_name": "propane",
        "inchi_key": None,
        "formula": "C3H8-a",
    }
    payload_b = {
        "key": "name:0003",
        "name": "propane",
        "normalized_name": "propane",
        "inchi_key": None,
        "formula": "C3H8-b",
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_concurrent_upsert,
                barrier,
                engine,
                payload_a,
                conflict_columns,
            ),
            executor.submit(
                run_concurrent_upsert,
                barrier,
                engine,
                payload_b,
                conflict_columns,
            ),
        ]
        for future in futures:
            future.result()

    with Session(engine) as session:
        rows = (
            session.execute(select(Adsorbate).where(Adsorbate.key == "name:0003"))
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].id is not None
        assert rows[0].formula in {"C3H8-a", "C3H8-b"}

    engine.dispose()
