from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from adsmod_common.config import DatabaseConfig
from adsmod_core.common.utils.logger import logger
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.schemas.models import Base


MIGRATION_LOCK_KEY = 8_174_209_531
MIGRATION_CONFIG_PATH = Path(__file__).resolve().parents[5] / "pyproject.toml"
INITIAL_REVISION = "23f1110c64a9"
CURRENT_TABLES = frozenset(Base.metadata.tables)


###############################################################################
class DatabaseMigrationError(RuntimeError):
    """Raised when the database cannot be brought to the packaged head."""


###############################################################################
class MigrationLockTimeoutError(DatabaseMigrationError):
    """Raised when another process holds the migration lock too long."""


###############################################################################
@dataclass(frozen=True)
class MigrationResult:
    backend: str
    before: tuple[str, ...]
    after: tuple[str, ...]
    head: str
    applied_migrations: bool


###############################################################################
def build_alembic_config() -> Config:
    if not MIGRATION_CONFIG_PATH.is_file():
        raise DatabaseMigrationError(
            f"Alembic configuration is missing: {MIGRATION_CONFIG_PATH}"
        )
    config = Config(toml_file=MIGRATION_CONFIG_PATH)
    script = ScriptDirectory.from_config(config)
    heads = tuple(script.get_heads())
    if len(heads) != 1:
        raise DatabaseMigrationError(
            "Alembic migration history must contain exactly one head; "
            f"found {len(heads)}. Merge migration branches before startup."
        )
    if script.get_revision(INITIAL_REVISION) is None:
        raise DatabaseMigrationError(
            f"Alembic initial revision {INITIAL_REVISION} is missing."
        )
    config.attributes["script_directory"] = script
    config.attributes["head_revision"] = heads[0]
    return config


###############################################################################
def _script_and_head(config: Config) -> tuple[ScriptDirectory, str]:
    script = config.attributes.get("script_directory")
    head = config.attributes.get("head_revision")
    if isinstance(script, ScriptDirectory) and isinstance(head, str):
        return script, head
    script = ScriptDirectory.from_config(config)
    heads = tuple(script.get_heads())
    if len(heads) != 1:
        raise DatabaseMigrationError(
            "Alembic migration history must contain exactly one head; "
            f"found {len(heads)}."
        )
    return script, heads[0]


###############################################################################
def _current_heads(connection: Connection) -> tuple[str, ...]:
    migration_context = MigrationContext.configure(
        connection,
        opts={"version_table": "alembic_version"},
    )
    return tuple(migration_context.get_current_heads())


###############################################################################
def _run_command(
    config: Config,
    connection: Connection,
    action: str,
    revision: str,
) -> None:
    config.attributes["connection"] = connection
    if action == "upgrade":
        command.upgrade(config, revision)
        return
    raise ValueError(f"Unsupported Alembic action: {action}")


###############################################################################
def _validate_known_heads(current: tuple[str, ...], script: ScriptDirectory) -> None:
    unknown = [
        revision for revision in current if script.get_revision(revision) is None
    ]
    if unknown:
        raise DatabaseMigrationError(
            "Database references revisions that are not packaged: "
            + ", ".join(sorted(unknown))
        )
    if len(current) > 1:
        raise DatabaseMigrationError(
            "Database contains multiple Alembic heads: " + ", ".join(sorted(current))
        )


###############################################################################
def _missing_current_tables(connection: Connection) -> set[str]:
    return CURRENT_TABLES - set(inspect(connection).get_table_names())


###############################################################################
def _migrate_locked(
    connection: Connection,
    config: Config,
    database: DatabaseConfig,
) -> MigrationResult:
    script, head = _script_and_head(config)
    table_names = set(inspect(connection).get_table_names())
    user_tables = table_names - {"alembic_version", "sqlite_sequence"}
    version_table_exists = "alembic_version" in table_names
    before = _current_heads(connection)
    _validate_known_heads(before, script)

    if not version_table_exists:
        if user_tables:
            raise DatabaseMigrationError(
                "Non-empty unversioned database detected; refusing to infer or stamp "
                "its schema. Export and re-import the data into a new versioned database."
            )
        logger.info("Database has no schema version; applying the Alembic history.")
    elif not before and user_tables:
        raise DatabaseMigrationError(
            "Database has an empty alembic_version table alongside application tables; "
            "it may contain an interrupted migration. Repair it manually."
        )

    backend = "sqlite" if database.embedded_database else "postgres"
    if before == (head,):
        missing = _missing_current_tables(connection)
        if missing:
            raise DatabaseMigrationError(
                "Database is stamped at Alembic head but is missing tables: "
                + ", ".join(sorted(missing))
            )
        return MigrationResult(
            backend=backend,
            before=before,
            after=before,
            head=head,
            applied_migrations=False,
        )

    before_label = ", ".join(before) if before else "base"
    logger.info("Applying Alembic migrations from %s to %s.", before_label, head)
    _run_command(config, connection, "upgrade", head)
    after = _current_heads(connection)
    _validate_known_heads(after, script)
    if after != (head,):
        raise DatabaseMigrationError(
            f"Alembic upgrade completed with current revisions {after!r}; expected {head!r}."
        )
    return MigrationResult(
        backend=backend,
        before=before,
        after=after,
        head=head,
        applied_migrations=True,
    )


###############################################################################
def _acquire_postgres_lock(connection: Connection, timeout_seconds: int) -> None:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while True:
        locked = connection.execute(
            text("SELECT pg_try_advisory_xact_lock(:lock_key)"),
            {"lock_key": MIGRATION_LOCK_KEY},
        ).scalar()
        if locked:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MigrationLockTimeoutError(
                f"Timed out after {timeout_seconds}s waiting for the PostgreSQL migration lock."
            )
        time.sleep(min(0.1, remaining))


###############################################################################
def migrate_engine(
    engine: Engine,
    database: DatabaseConfig,
    *,
    storage_root: Path | None = None,
    config: Config | None = None,
) -> MigrationResult:
    del storage_root
    alembic_config = config or build_alembic_config()
    backend = "sqlite" if database.embedded_database else "postgres"
    started = time.perf_counter()
    logger.info("Checking %s database schema and Alembic revision.", backend)

    if backend == "sqlite":
        with engine.connect() as connection:
            driver_connection = connection.connection.driver_connection
            driver_connection.autocommit = True
            try:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            except OperationalError as exc:
                connection.rollback()
                driver_connection.autocommit = False
                driver_connection.rollback()
                raise MigrationLockTimeoutError(
                    "Timed out waiting for the SQLite migration write lock."
                ) from exc
            try:
                result = _migrate_locked(connection, alembic_config, database)
            except Exception:
                connection.exec_driver_sql("ROLLBACK")
                connection.rollback()
                driver_connection.autocommit = False
                driver_connection.rollback()
                raise
            else:
                try:
                    connection.exec_driver_sql("COMMIT")
                    connection.commit()
                finally:
                    driver_connection.autocommit = False
                    driver_connection.rollback()
    else:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                _acquire_postgres_lock(connection, database.connect_timeout)
                result = _migrate_locked(connection, alembic_config, database)
            except Exception:
                transaction.rollback()
                raise
            else:
                transaction.commit()

    elapsed = time.perf_counter() - started
    logger.info(
        "Database migration status=success backend=%s before=%s after=%s head=%s "
        "applied_migrations=%s lock_timeout=%ss elapsed=%.2fs.",
        backend,
        ",".join(result.before) or "base",
        ",".join(result.after) or "base",
        result.head,
        result.applied_migrations,
        database.connect_timeout,
        elapsed,
    )
    return result


###############################################################################
def migrate_database(
    database: DatabaseConfig,
    *,
    storage_root: Path | None = None,
) -> MigrationResult:
    manager: DatabaseManager | None = None
    try:
        manager = DatabaseManager(database, storage_root=storage_root)
        return migrate_engine(
            manager.engine,
            database,
            storage_root=storage_root,
        )
    except DatabaseMigrationError:
        raise
    except (SQLAlchemyError, OSError, ValueError) as exc:
        raise DatabaseMigrationError(
            "Database migration failed; verify database connectivity and schema state."
        ) from exc
    finally:
        if manager is not None:
            manager.dispose()


__all__ = [
    "DatabaseMigrationError",
    "MigrationLockTimeoutError",
    "MigrationResult",
    "build_alembic_config",
    "migrate_database",
    "migrate_engine",
]
