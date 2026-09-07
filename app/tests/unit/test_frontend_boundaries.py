from __future__ import annotations

from pathlib import Path


###############################################################################
def test_training_routes_are_capability_guarded() -> None:
    text = Path("app/client/src/app/app.routes.ts").read_text(encoding="utf-8")
    assert "machineLearningGuard" in text
    assert "machineLearningEntryGuard" in text
    assert "path: 'training'" in text
    assert "path: 'training/:view'" in text
    assert text.count("canActivate: [machineLearningEntryGuard]") == 1
    assert text.count("canActivate: [machineLearningGuard]") == 1


###############################################################################
def test_frontend_uses_one_backend_proxy() -> None:
    text = Path("app/client/proxy.conf.cjs").read_text(encoding="utf-8")
    assert "'/api/v1'" in text
    assert "'/health'" in text
    assert "'/api/v1/training'" not in text
    assert "ml-health" not in text
    assert "ml_port" not in text


###############################################################################
def test_capabilities_are_centralized() -> None:
    service = Path("app/client/src/app/services/system.service.ts").read_text(encoding="utf-8")
    shell = Path("app/client/src/app/layout/core-shell.component.ts").read_text(encoding="utf-8")
    assert "fetchApplicationCapabilities" in service
    assert "features.machine_learning" in service
    assert "fetchMlReadiness" not in service
    assert "ML Service" not in shell
