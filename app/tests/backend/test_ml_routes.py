from __future__ import annotations

from pathlib import Path
from fastapi.testclient import TestClient
from adsmod_common.config import StorageConfig, load_config
from adsmod_core.app import create_app

CONFIG_PATH = Path("app/resources/adsmod.json")


###############################################################################
def _config(tmp_path: Path):
    base = load_config(CONFIG_PATH)
    return base.model_copy(update={
        "storage": StorageConfig(root=tmp_path),
        "application": base.application.model_copy(update={
            "database": base.application.database.model_copy(update={"sqlite_path": "data/test.db"})
        }),
    })


###############################################################################
def test_ml_routes_are_mounted_on_the_single_backend(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.get("/health/live").json()["service"] == "backend"
        assert client.get("/api/v1/training/configuration").status_code == 200
        assert client.get("/api/v1/training/status").status_code == 200
        assert client.get("/api/v1/system/configuration").status_code == 200


###############################################################################
def test_standalone_ml_server_entrypoints_are_removed() -> None:
    root = Path("app/server/ml/src/adsmod_ml")
    assert not (root / "app.py").exists()
    assert not (root / "cli.py").exists()
    assert not (root / "http" / "entrypoint.py").exists()
    assert not (root / "clients" / "core_client.py").exists()
