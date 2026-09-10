import time
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fastapi.testclient import TestClient
from adsmod_common.config import StorageConfig, load_config
from adsmod_core import app as app_module
from adsmod_core.app import create_app, create_app_from_path

CONFIG_PATH = Path("app/resources/adsmod.json")


###############################################################################
def _shutdown_process_runner(stop_event: Any) -> dict[str, str]:
    while not stop_event.is_set():
        time.sleep(0.01)
    return {"stopped": "true"}


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
def test_optional_ml_missing_package_is_reported_as_unavailable(
    monkeypatch, caplog  # type: ignore[no-untyped-def]
) -> None:
    def missing_package(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name="adsmod_ml")

    monkeypatch.setattr(app_module, "import_module", missing_package)
    runtime = SimpleNamespace(
        config=load_config(CONFIG_PATH),
        training_data=None,
        machine_learning_available=True,
        machine_learning_reason=None,
    )

    app_module._register_optional_ml(object(), runtime)

    assert runtime.machine_learning_available is False
    assert "not installed" in caplog.text


###############################################################################
def test_optional_ml_bootstrap_failure_is_logged_as_initialization_error(
    monkeypatch, caplog  # type: ignore[no-untyped-def]
) -> None:
    def broken_package(name: str):
        raise RuntimeError(f"broken optional package at {name}")

    monkeypatch.setattr(app_module, "import_module", broken_package)
    runtime = SimpleNamespace(
        config=load_config(CONFIG_PATH),
        training_data=None,
        machine_learning_available=True,
        machine_learning_reason=None,
    )

    app_module._register_optional_ml(object(), runtime)

    assert runtime.machine_learning_available is False
    assert "initialization failed" in caplog.text


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


###############################################################################
def test_in_process_snapshot_access_reconstructs_multiple_pages(
    tmp_path: Path,
) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        with TestClient(create_app(_temporary_config(directory))) as client:
            access = client.app.state.runtime.training_data
            rows = [
                {"id": index, "value": f"value-{index}"}
                for index in range(1001)
            ]
            reference = access.create_snapshot(rows)

            payload = access.fetch_snapshot(reference.snapshot_id)

            assert payload.snapshot_id == reference.snapshot_id
            assert payload.content_hash == reference.content_hash
            assert payload.rows == tuple(rows)


###############################################################################
def test_lifespan_stops_active_process_jobs(tmp_path: Path) -> None:
    with TemporaryDirectory(dir=tmp_path) as directory:
        manager = None
        worker = None
        with TestClient(create_app(_temporary_config(directory))) as client:
            manager = client.app.state.core_container.job_manager
            job_id = manager.start_job(
                "shutdown-test",
                _shutdown_process_runner,
                run_mode="process",
            )
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                process_state = manager.processes.get(job_id)
                worker = process_state.process if process_state is not None else None
                if worker is not None and worker.is_alive():
                    break
                status = manager.get_job_status(job_id)
                if status and status["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
            assert worker is not None
            assert worker.is_alive()

        assert manager is not None
        if worker is not None:
            worker.join(timeout=3.0)
            assert not worker.is_alive()
        assert manager.processes == {}
        assert manager.threads == {}
