from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from adsmod_common.config import DatabaseConfig
from adsmod_core.common.utils.logger import logger
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.database.migrator import (
    CURRENT_TABLES,
    DatabaseMigrationError,
    MIGRATION_LOCK_KEY,
    MigrationLockTimeoutError,
    MigrationResult,
    migrate_database,
)
from adsmod_core.repositories.database.sql import (
    build_postgres_create_database_sql,
    build_postgres_database_exists_sql,
)


###############################################################################
def clone_database_config_with_name(
    database: DatabaseConfig,
    database_name: str,
) -> DatabaseConfig:
    return database.model_copy(
        update={
            "embedded_database": False,
            "database_name": database_name,
        }
    )


###############################################################################
def _validate_postgres_settings(database: DatabaseConfig) -> None:
    if not database.host or not database.username or not database.database_name:
        raise ValueError("PostgreSQL host, database name, and username are required.")


###############################################################################
def _acquire_creation_lock(connection: Connection, timeout_seconds: int) -> None:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while True:
        locked = connection.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": MIGRATION_LOCK_KEY},
        ).scalar()
        if locked:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MigrationLockTimeoutError(
                f"Timed out after {timeout_seconds}s waiting for the PostgreSQL database lock."
            )
        time.sleep(min(0.1, remaining))


###############################################################################
def _ensure_postgres_database_exists(database: DatabaseConfig) -> None:
    _validate_postgres_settings(database)
    admin_manager: DatabaseManager | None = None
    try:
        admin_database = clone_database_config_with_name(database, "postgres")
        admin_manager = DatabaseManager(admin_database)
        admin_engine = admin_manager.engine.execution_options(
            isolation_level="AUTOCOMMIT"
        )
        with admin_engine.connect() as connection:
            _acquire_creation_lock(connection, database.connect_timeout)
            try:
                exists = connection.execute(
                    build_postgres_database_exists_sql(),
                    {"name": database.database_name},
                ).scalar()
                if not exists:
                    logger.info(
                        "Creating PostgreSQL database %s before running Alembic.",
                        database.database_name,
                    )
                    connection.execute(
                        build_postgres_create_database_sql(database.database_name)
                    )
            finally:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_key)"),
                    {"lock_key": MIGRATION_LOCK_KEY},
                )
    except DatabaseMigrationError:
        raise
    except SQLAlchemyError as exc:
        detail = str(getattr(exc, "orig", exc)).lower()
        if "permission denied" in detail and "database" in detail:
            raise DatabaseMigrationError(
                "The configured PostgreSQL role lacks CREATEDB permission; "
                "grant CREATEDB or pre-create the target database."
            ) from exc
        raise DatabaseMigrationError(
            "PostgreSQL database creation failed; verify the configured host, "
            "credentials, and CREATEDB permission."
        ) from exc
    except ValueError as exc:
        raise DatabaseMigrationError(
            "PostgreSQL database creation failed; verify the configured host, "
            "credentials, and CREATEDB permission."
        ) from exc
    finally:
        if admin_manager is not None:
            admin_manager.dispose()


###############################################################################
def verify_postgres_database(database: DatabaseConfig) -> None:
    """Perform a non-mutating connectivity and current-schema probe."""

    _validate_postgres_settings(database)
    manager: DatabaseManager | None = None
    try:
        manager = DatabaseManager(database)
        with manager.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            tables = set(inspect(connection).get_table_names())
        if not CURRENT_TABLES.issubset(tables):
            raise RuntimeError(
                "PostgreSQL schema is incomplete; run database initialization."
            )
    except (SQLAlchemyError, ValueError) as exc:
        logger.error("PostgreSQL readiness probe failed (%s).", type(exc).__name__)
        raise RuntimeError(
            "PostgreSQL is unavailable or not initialized; inspect database settings."
        ) from None
    finally:
        if manager is not None:
            manager.dispose()


###############################################################################
def initialize_sqlite_database(
    database: DatabaseConfig,
    *,
    storage_root: Path | None = None,
) -> MigrationResult:
    return migrate_database(database, storage_root=storage_root)


###############################################################################
def initialize_postgres_database(database: DatabaseConfig) -> MigrationResult:
    _ensure_postgres_database_exists(database)
    return migrate_database(database)


###############################################################################
def prepare_database_for_startup(
    database: DatabaseConfig,
    *,
    storage_root: Path | None = None,
) -> MigrationResult:
    if database.embedded_database:
        return initialize_sqlite_database(database, storage_root=storage_root)
    return initialize_postgres_database(database)


###############################################################################
def run_database_initialization(
    database: DatabaseConfig,
    *,
    storage_root: Path | None = None,
) -> MigrationResult:
    return prepare_database_for_startup(database, storage_root=storage_root)


###############################################################################
def initialize_database(
    database: DatabaseConfig,
    *,
    storage_root: Path | None = None,
) -> MigrationResult:
    try:
        return run_database_initialization(database, storage_root=storage_root)
    except DatabaseMigrationError as exc:
        logger.error("Database initialization failed: %s", exc)
        raise RuntimeError(str(exc)) from None
    except (SQLAlchemyError, OSError, ValueError) as exc:
        logger.error("Database initialization failed (%s).", type(exc).__name__)
        raise RuntimeError(
            "Database initialization failed; verify the configured database is reachable."
        ) from None


__all__ = [
    "clone_database_config_with_name",
    "initialize_database",
    "initialize_postgres_database",
    "initialize_sqlite_database",
    "prepare_database_for_startup",
    "run_database_initialization",
    "verify_postgres_database",
]
