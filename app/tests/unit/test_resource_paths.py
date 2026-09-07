from pathlib import Path

from adsmod_common.config import DatabaseConfig, StorageConfig, load_config
from adsmod_core.repositories.database.manager import resolve_sqlite_path
from adsmod_core.persistence.paths import resolve_database_path

###############################################################################
def test_relative_database_path_follows_the_canonical_storage_root(
    tmp_path: Path,
) -> None:
    database = DatabaseConfig(embedded_database=True, sqlite_path="data/database.db")

    assert (
        resolve_sqlite_path(database, storage_root=tmp_path)
        == (tmp_path / "data" / "database.db").resolve()
    )

    config = load_config(Path("app/resources/adsmod.json")).model_copy(
        update={"storage": StorageConfig(root=tmp_path)}
    )
    assert (
        resolve_database_path(config) == (tmp_path / "data" / "database.db").resolve()
    )
