from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from adsmod_common.config import DatabaseConfig
from adsmod_core.common.utils.logger import logger
from adsmod_core.repositories.database.utils import normalize_postgres_engine


###############################################################################
def _expand_path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    return Path(expanded)


###############################################################################
def resolve_sqlite_path(
    database: DatabaseConfig,
    *,
    storage_root: Path | None = None,
) -> Path:
    """Resolve the configured SQLite file against the canonical storage root."""

    if not database.sqlite_path:
        raise ValueError("application.database.sqlite_path is required for SQLite")
    if database.sqlite_path == ":memory:":
        return Path(database.sqlite_path)

    configured_path = _expand_path(database.sqlite_path)
    if configured_path.is_absolute():
        return configured_path.resolve()
    base = storage_root.resolve() if storage_root is not None else Path.cwd().resolve()
    return (base / configured_path).resolve()


###############################################################################
class DatabaseManager:
    """Own Core's one engine, session factory, and transaction boundary."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        database: DatabaseConfig,
        *,
        storage_root: Path | None = None,
    ) -> None:
        self.database = database
        self.storage_root = storage_root
        self.backend = (
            "sqlite"
            if database.embedded_database
            else self._normalize_backend(database.engine)
        )
        self.engine = self._create_engine()
        self.session_factory = sessionmaker(
            bind=self.engine,
            future=True,
            expire_on_commit=False,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_backend(engine: str | None) -> str:
        value = (engine or "postgres").strip().lower()
        if value in {"postgres", "postgresql", "postgresql+psycopg"}:
            return "postgres"
        raise ValueError(f"Unsupported database engine: {engine}")

    # -------------------------------------------------------------------------
    def _create_engine(self) -> Engine:
        if self.backend == "sqlite":
            sqlite_connect_args: dict[str, Any] = {
                "check_same_thread": False,
                "autocommit": False,
            }
            if self.database.sqlite_path == ":memory:":
                engine = create_engine(
                    "sqlite:///:memory:",
                    future=True,
                    connect_args=sqlite_connect_args,
                )
                event.listen(engine, "connect", self._configure_sqlite)
                return engine

            path = resolve_sqlite_path(self.database, storage_root=self.storage_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            sqlite_connect_args["timeout"] = self.database.connect_timeout
            engine = create_engine(
                f"sqlite:///{path}",
                future=True,
                connect_args=sqlite_connect_args,
            )
            event.listen(engine, "connect", self._configure_sqlite)
            return engine

        if (
            not self.database.host
            or not self.database.database_name
            or not self.database.username
        ):
            raise ValueError(
                "PostgreSQL host, database name, and username are required."
            )
        engine_name = normalize_postgres_engine(self.database.engine)
        username = quote_plus(self.database.username)
        password = quote_plus(self.database.password or "")
        url = (
            f"{engine_name}://{username}:{password}@{self.database.host}:"
            f"{self.database.port}/{self.database.database_name}"
        )
        connect_args: dict[str, Any] = {
            "connect_timeout": self.database.connect_timeout,
            "client_encoding": "utf8",
        }
        if self.database.ssl:
            connect_args["sslmode"] = "require"
            if self.database.ssl_ca:
                connect_args["sslrootcert"] = self.database.ssl_ca
        return create_engine(
            url, future=True, connect_args=connect_args, pool_pre_ping=True
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _configure_sqlite(dbapi_connection: Any, connection_record: Any) -> None:
        del connection_record
        previous_autocommit = getattr(dbapi_connection, "autocommit", None)
        if previous_autocommit is not None:
            dbapi_connection.autocommit = True
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()
            if previous_autocommit is not None:
                dbapi_connection.autocommit = previous_autocommit
                dbapi_connection.rollback()

    # -------------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self.session_factory() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    # -------------------------------------------------------------------------
    def session(self) -> Session:
        return self.session_factory()

    # -------------------------------------------------------------------------
    def dispose(self) -> None:
        logger.debug("Disposing %s database engine", self.backend)
        self.engine.dispose()


__all__ = ["DatabaseManager", "resolve_sqlite_path"]
