from __future__ import annotations

import json
from pathlib import Path

OPENAPI_PATH = Path("app/server/openapi/backend.json")


###############################################################################
def test_unified_openapi_snapshot_covers_complete_surface() -> None:
    schema = json.loads(OPENAPI_PATH.read_text(encoding="utf-8"))
    assert schema["openapi"] == "3.1.0"
    paths = schema["paths"]
    assert "/api/v1/datasets" in paths
    assert "/api/v1/fitting/models" in paths
    assert "/api/v1/training/datasets" in paths
    assert "/api/v1/training/configuration" in paths
