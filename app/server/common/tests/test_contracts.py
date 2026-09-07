import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from adsmod_common.capabilities import CapabilitiesResponse
from adsmod_common.config import AdsmodConfig, load_config

CONFIG_PATH = Path("app/resources/adsmod.json")


###############################################################################
def test_canonical_config_loads_single_backend_runtime() -> None:
    config = load_config(CONFIG_PATH)
    assert config.version == "3.0.0"
    assert config.runtime.backend_port != config.runtime.frontend_port
    assert not hasattr(config.runtime, "mode")
    assert not hasattr(config.runtime, "ml_port")


###############################################################################
def test_legacy_dual_backend_runtime_keys_are_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["mode"] = "core-ml"
    with pytest.raises(ValidationError):
        AdsmodConfig.model_validate(payload)


###############################################################################
def test_duplicate_backend_and_frontend_ports_are_rejected() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["runtime"]["frontend_port"] = payload["runtime"]["backend_port"]
    with pytest.raises(ValidationError):
        AdsmodConfig.model_validate(payload)


###############################################################################
def test_capability_contract_is_strict() -> None:
    response = CapabilitiesResponse.model_validate({
        "version": "3.0.0",
        "features": {
            "datasets": True,
            "nist": True,
            "fitting": True,
            "machine_learning": False,
            "training": False,
            "checkpoints": False,
        },
    })
    assert response.features.machine_learning is False
