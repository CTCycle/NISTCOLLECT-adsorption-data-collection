from __future__ import annotations

from alembic import command
from sqlalchemy import inspect

from adsmod_common.config import DatabaseConfig
from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.database.migrator import (
    build_alembic_config,
    migrate_engine,
)
from adsmod_core.repositories.schemas.models import Base

###############################################################################
def _sqlite_settings(path: str) -> DatabaseConfig:
    return DatabaseConfig(
        embedded_database=True,
        connect_timeout=5,
        insert_batch_size=100,
        sqlite_path=path,
    )

###############################################################################
def test_packaged_history_has_one_head_and_no_pending_operations(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = _sqlite_settings(str(tmp_path / "quality.db"))
    manager = DatabaseManager(settings)
    try:
        result = migrate_engine(manager.engine, settings)
        config = build_alembic_config()
        with manager.engine.connect() as connection:
            config.attributes["connection"] = connection
            command.check(config)
            assert set(inspect(connection).get_table_names()) >= set(
                Base.metadata.tables
            )
        assert result.after == (result.head,)
        assert config.attributes["script_directory"].get_heads() == [result.head]
    finally:
        manager.dispose()
