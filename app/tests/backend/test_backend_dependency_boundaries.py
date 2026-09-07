from __future__ import annotations

import re
from pathlib import Path

BACKEND_ROOT = Path("app/server")
GENERATED_DIRS = {".venv", "__pycache__", ".pytest_cache", ".uv-cache"}


###############################################################################
def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        if not any(part in GENERATED_DIRS for part in path.parts):
            yield path


###############################################################################
def _has_import(text: str, package: str) -> bool:
    return bool(re.search(rf"^(?:from|import) {re.escape(package)}(?:\.|\s|$)", text, re.MULTILINE))


###############################################################################
def test_common_has_no_framework_or_persistence_imports() -> None:
    for path in _iter_python_files(BACKEND_ROOT / "common"):
        text = path.read_text(encoding="utf-8")
        hits = [pkg for pkg in ("fastapi", "sqlalchemy") if _has_import(text, pkg)]
        assert not hits, f"{path}: forbidden imports {hits}"


###############################################################################
def test_core_does_not_statically_import_heavy_ml_packages() -> None:
    for path in _iter_python_files(BACKEND_ROOT / "core"):
        text = path.read_text(encoding="utf-8")
        hits = [pkg for pkg in ("torch", "keras", "sklearn") if _has_import(text, pkg)]
        assert not hits, f"{path}: forbidden imports {hits}"


###############################################################################
def test_ml_extension_does_not_own_persistence_layer() -> None:
    for path in _iter_python_files(BACKEND_ROOT / "ml"):
        text = path.read_text(encoding="utf-8")
        hits = [pkg for pkg in ("sqlalchemy", "alembic") if _has_import(text, pkg)]
        assert not hits, f"{path}: forbidden imports {hits}"


###############################################################################
def test_contract_packages_have_no_framework_or_persistence_imports() -> None:
    for package_root in (
        BACKEND_ROOT / "common" / "src",
        BACKEND_ROOT / "core" / "src" / "adsmod_core" / "contracts",
        BACKEND_ROOT / "ml" / "src" / "adsmod_ml" / "contracts",
    ):
        for path in _iter_python_files(package_root):
            text = path.read_text(encoding="utf-8")
            hits = [pkg for pkg in ("fastapi", "sqlalchemy") if _has_import(text, pkg)]
            assert not hits, f"{path}: forbidden imports {hits}"
