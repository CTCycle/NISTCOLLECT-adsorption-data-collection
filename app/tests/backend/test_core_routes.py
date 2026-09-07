from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from adsmod_common.config import StorageConfig, load_config
from adsmod_core.app import create_app

CONFIG_PATH = Path("app/resources/adsmod.json")


###############################################################################
def _config(tmp_path: Path):  # type: ignore[no-untyped-def]
    base = load_config(CONFIG_PATH)
    return base.model_copy(
        update={
            "storage": StorageConfig(root=tmp_path),
            "application": base.application.model_copy(
                update={
                    "database": base.application.database.model_copy(
                        update={"sqlite_path": "data/test.db"}
                    )
                }
            ),
        }
    )


###############################################################################
def test_unified_routes_expose_core_optional_ml_and_public_data_surface(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        assert client.get("/health/live").json()["service"] == "backend"
        assert client.get("/health/ready").json()["state"] == "ready"
        capabilities = client.get("/api/v1/system/capabilities").json()
        assert capabilities["features"]["datasets"] is True
        assert capabilities["features"]["machine_learning"] is True, client.app.state.runtime.machine_learning_reason
        assert client.get("/api/v1/datasets").status_code == 200
        assert client.get("/api/v1/training/configuration").status_code == 200
        sources = client.get("/api/v1/public-data/sources?check_health=false")
        assert sources.status_code == 200
        assert {item["key"] for item in sources.json()["sources"]} == {
            "nist",
            "pubchem",
            "cod",
        }
        assert client.get("/api/health").status_code == 404


###############################################################################
def test_unified_openapi_surface_contains_public_data_routes(tmp_path: Path) -> None:
    with TestClient(create_app(_config(tmp_path))) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/datasets" in paths
        assert "/api/v1/fitting/models" in paths
        assert "/api/v1/training/configuration" in paths
        assert "/api/v1/public-data/sources" in paths
        assert "/api/v1/public-data/adsorption" in paths
        assert "/api/v1/public-data/chemicals/resolve" in paths
        assert "/api/v1/public-data/structures/search" in paths
