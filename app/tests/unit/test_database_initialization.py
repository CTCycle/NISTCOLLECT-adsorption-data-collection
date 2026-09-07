from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy.exc import SQLAlchemyError

from adsmod_common.config import DatabaseConfig
from adsmod_core.repositories.database import initializer, migrator
from adsmod_core.repositories.database.initializer import (
    DatabaseMigrationError,
    MigrationLockTimeoutError,
)
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.schemas.models import Base


###############################################################################
def sqlite_config(path: Path, *, timeout: int = 5) -> DatabaseConfig:
    return DatabaseConfig(
        embedded_database=True,
        connect_timeout=timeout,
        insert_batch_size=100,
        sqlite_path=str(path),
    )


###############################################################################
def table_names(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()


###############################################################################
def test_sqlite_missing_database_runs_baseline_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "database.db"
    database = sqlite_config(path)

    first = initializer.prepare_database_for_startup(database)
    second = initializer.initialize_database(database)

    assert first.applied_migrations is True
    assert second.applied_migrations is False
    assert "alembic_version" in table_names(path)
    assert len(table_names(path) & set(Base.metadata.tables)) == len(Base.metadata.tables)


###############################################################################
def test_sqlite_empty_existing_file_is_initialized(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    path.touch()

    result = migrator.migrate_database(sqlite_config(path))

    assert result.after == (result.head,)
    assert table_names(path) >= {"alembic_version", "datasets"}


###############################################################################
def test_nonempty_unversioned_database_is_rejected_without_inference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unversioned.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    with pytest.raises(DatabaseMigrationError, match="Non-empty unversioned"):
        migrator.migrate_database(sqlite_config(path))

    assert table_names(path) == {"unrelated"}


###############################################################################
def test_empty_version_table_with_application_tables_fails_safely(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interrupted.db"
    manager = DatabaseManager(sqlite_config(path))
    try:
        Base.metadata.create_all(manager.engine)
    finally:
        manager.dispose()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        connection.commit()

    with pytest.raises(DatabaseMigrationError, match="empty alembic_version"):
        migrator.migrate_database(sqlite_config(path))


###############################################################################
def test_failed_migration_rolls_back_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "rollback.db"
    database = sqlite_config(path)
    config = migrator.build_alembic_config()

    def fail_after_ddl(_config, connection, _action, _revision):  # type: ignore[no-untyped-def]
        connection.exec_driver_sql("CREATE TABLE transient_failure (id INTEGER)")
        raise SQLAlchemyError("synthetic migration failure")

    monkeypatch.setattr(migrator, "_run_command", fail_after_ddl)
    manager = DatabaseManager(database)
    try:
        with pytest.raises(SQLAlchemyError, match="synthetic"):
            migrator.migrate_engine(manager.engine, database, config=config)
    finally:
        manager.dispose()

    assert "transient_failure" not in table_names(path)
    assert "alembic_version" not in table_names(path)


###############################################################################
def test_concurrent_sqlite_startup_serializes_migrations(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.db"

    def start_once():  # type: ignore[no-untyped-def]
        return migrator.migrate_database(sqlite_config(path))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: start_once(), range(2)))

    assert all(result.after == (result.head,) for result in results)
    assert table_names(path) >= {"alembic_version", "datasets"}


###############################################################################
def test_sqlite_migration_lock_timeout_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "locked.db"
    blocker = sqlite3.connect(path, timeout=5)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(MigrationLockTimeoutError):
            migrator.migrate_database(sqlite_config(path, timeout=1))
    finally:
        blocker.rollback()
        blocker.close()


###############################################################################
def test_database_initialization_sanitizes_sqlalchemy_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def raise_database_error(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise SQLAlchemyError("postgresql://postgres:secret@host/database")

    monkeypatch.setattr(
        initializer, "run_database_initialization", raise_database_error
    )

    with pytest.raises(RuntimeError) as error:
        initializer.initialize_database(sqlite_config(tmp_path / "unused.db"))

    assert "secret" not in str(error.value)
    assert "verify" in str(error.value)


###############################################################################
def test_postgres_creation_permission_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = DatabaseConfig(
        embedded_database=False,
        engine="postgres",
        host="localhost",
        port=5432,
        database_name="adsmod_permission_test",
        username="app",
        password="secret",
        ssl=False,
        connect_timeout=1,
        insert_batch_size=100,
        sqlite_path=None,
    )

    def fail_to_connect(_database: DatabaseConfig) -> DatabaseManager:
        raise SQLAlchemyError("permission denied to create database")

    monkeypatch.setattr(initializer, "DatabaseManager", fail_to_connect)
    with pytest.raises(DatabaseMigrationError, match="CREATEDB"):
        initializer._ensure_postgres_database_exists(database)
