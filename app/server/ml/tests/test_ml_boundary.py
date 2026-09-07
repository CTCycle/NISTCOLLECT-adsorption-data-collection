from __future__ import annotations

import inspect
from pathlib import Path

from adsmod_ml.services.container import MlServiceContainer


###############################################################################
def test_ml_container_consumes_in_process_snapshot_access() -> None:
    parameters = inspect.signature(MlServiceContainer).parameters
    assert "snapshot_access" in parameters
    assert "internal_token" not in parameters


###############################################################################
def test_ml_extension_has_no_standalone_fastapi_server_or_core_http_client() -> None:
    root = Path("app/server/ml/src/adsmod_ml")
    assert not (root / "app.py").exists()
    assert not (root / "cli.py").exists()
    assert not (root / "http" / "entrypoint.py").exists()
    assert not (root / "clients" / "core_client.py").exists()


###############################################################################
def test_ml_source_has_no_backend_to_backend_http_boundary() -> None:
    root = Path("app/server/ml/src/adsmod_ml")
    combined = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert "CoreSnapshotClient" not in combined
    assert "core_base_url" not in combined
    assert "X-ADSMOD-Internal-Token" not in combined
