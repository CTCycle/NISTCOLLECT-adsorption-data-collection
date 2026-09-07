from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from adsmod_common.config import DatabaseConfig
from adsmod_core.repositories.database.bulk import upsert_records
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.schemas import Base
from adsmod_core.repositories.schemas.models import Dataset
from adsmod_core.repositories.schemas.types import JSONList, UTCDateTime


EXPECTED_TABLES = {
    "datasets",
    "adsorbates",
    "adsorbents",
    "isotherms",
    "isotherm_components",
    "observations",
    "dataset_imports",
    "fitting_runs",
    "fit_results",
    "fit_parameters",
    "training_snapshots",
    "training_snapshot_rows",
    "data_sources",
    "source_records",
    "adsorbate_source_records",
    "adsorbent_source_records",
    "isotherm_source_records",
    "structures",
    "structure_source_records",
    "structure_atoms",
    "adsorbate_synonyms",
    "chemical_properties",
    "material_properties",
    "references",
    "source_record_references",
}

###############################################################################
@pytest.mark.parametrize(
    "engine",
    ["postgres", "postgresql", "postgresql+psycopg"],
)
def test_manager_accepts_supported_postgres_engines(engine: str) -> None:
    assert DatabaseManager._normalize_backend(engine) == "postgres"

###############################################################################
def test_manager_rejects_removed_psycopg2_engine() -> None:
    with pytest.raises(ValueError, match="Unsupported database engine"):
        DatabaseManager._normalize_backend("postgresql+psycopg2")

###############################################################################
def test_canonical_schema_has_only_expected_tables() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES
    assert isinstance(Dataset.__table__.c.tags.type, JSONList)
    assert isinstance(Dataset.__table__.c.created_at.type, UTCDateTime)

###############################################################################
@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect()])
def test_every_canonical_table_compiles_for_both_backends(dialect) -> None:  # type: ignore[no-untyped-def]
    for table in Base.metadata.sorted_tables:
        CreateTable(table).compile(dialect=dialect)

###############################################################################
def test_manager_enables_sqlite_integrity_and_rolls_back() -> None:
    settings = DatabaseConfig(
        embedded_database=True,
        connect_timeout=30,
        insert_batch_size=100,
        sqlite_path=":memory:",
    )
    manager = DatabaseManager(settings)
    try:
        Base.metadata.create_all(manager.engine)
        with manager.transaction() as session:
            session.add(
                Dataset(
                    name="Water",
                    source="uploaded",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )
        with pytest.raises(IntegrityError):
            with manager.transaction() as session:
                session.add(Dataset(name=" water ", source="uploaded"))
        with manager.transaction() as session:
            assert session.execute(select(Dataset.name)).scalars().all() == ["Water"]
    finally:
        manager.dispose()

###############################################################################
def test_explicit_bulk_upsert_uses_declared_conflict_key() -> None:
    settings = DatabaseConfig(
        embedded_database=True,
        connect_timeout=30,
        insert_batch_size=100,
        sqlite_path=":memory:",
    )
    manager = DatabaseManager(settings)
    try:
        Base.metadata.create_all(manager.engine)
        with manager.transaction() as session:
            assert (
                upsert_records(
                    session,
                    Dataset.__table__,
                    [
                        {
                            "name": "A",
                            "normalized_name": "a",
                            "source": "uploaded",
                            "description": "one",
                            "tags": [],
                        }
                    ],
                    ["normalized_name"],
                )
                == 1
            )
        with manager.transaction() as session:
            upsert_records(
                session,
                Dataset.__table__,
                [
                    {
                        "name": "A2",
                        "normalized_name": "a",
                        "source": "uploaded",
                        "description": "two",
                        "tags": [],
                    }
                ],
                ["normalized_name"],
            )
        with manager.transaction() as session:
            row = session.execute(select(Dataset)).scalar_one()
            assert row.name == "A2"
            assert row.description == "two"
    finally:
        manager.dispose()
