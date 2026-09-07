from __future__ import annotations

import importlib
import os

###############################################################################
def test_ml_bootstrap_configures_torch_backend_before_keras_import(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    bootstrap = importlib.import_module("adsmod_ml.bootstrap")
    importlib.reload(bootstrap)
    bootstrap.configure_environment()
    assert os.environ["KERAS_BACKEND"] == "torch"
    assert os.environ["MPLBACKEND"] == "Agg"
