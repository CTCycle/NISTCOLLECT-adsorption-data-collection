from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from adsmod_common.config import StorageConfig, load_config
from adsmod_core.app import create_app, create_app_from_path

CONFIG_PATH = Path("app/resources/adsmod.json")


###############################################################################
def _temporary_config(directory: str):
    base = load_config(CONFIG_PATH)
    return base.model_copy(update={
        "storage": StorageConfig(root=Path(directory)),
        "application": base.application.model_copy(update={
            "database": base.application.database.model_copy(update={"sqlite_path": "core.db"})
        }),
    })


###############################################################################
def test_unified_runtime_contracts(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        with TestClient(create_app(_temporary_config(directory))) as client:
            assert client.get("/health/live").json()["service"] == "backend"
            assert client.get("/health/ready").json()["state"] == "ready"
            capabilities = client.get("/api/v1/system/capabilities").json()
            assert capabilities["features"]["datasets"] is True
            assert capabilities["features"]["machine_learning"] is True, client.app.state.runtime.machine_learning_reason
            assert client.get("/api/v1/system/configuration").status_code == 200
            assert client.get("/api/v1/training/configuration").status_code == 200


###############################################################################
def test_factory_uses_single_backend_port() -> None:
    application = create_app_from_path(CONFIG_PATH)
    assert application.state.config.runtime.backend_port > 0
    assert not hasattr(application.state.config.runtime, "ml_port")


###############################################################################
def test_in_process_snapshot_access_preserves_hash(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        with TestClient(create_app(_temporary_config(directory))) as client:
            access = client.app.state.runtime.training_data
            rows = [{"id": 1, "value": "alpha"}, {"id": 2, "value": "beta"}]
            reference = access.create_snapshot(rows)
            rows[0]["value"] = "mutated-after-create"
            payload = access.fetch_snapshot(reference.snapshot_id)
            assert payload.rows[0] == {"id": 1, "value": "alpha"}
            assert payload.content_hash == reference.content_hash
