"""Pytest configuration and shared fixtures for ADSMOD E2E tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from playwright.sync_api import APIRequestContext, Page, Playwright

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
APP_ROOT = TESTS_DIR.parent
WILDCARD_BIND_HOSTS = {"", "0.0.0.0", "::", "[::]"}
CANONICAL_CONFIG = APP_ROOT / "resources" / "adsmod.json"


###############################################################################
def normalize_client_host(bind_host: str) -> str:
    stripped = bind_host.strip()
    return "127.0.0.1" if stripped in WILDCARD_BIND_HOSTS else stripped


###############################################################################
def resolve_test_urls() -> tuple[str, str]:
    runtime = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))["runtime"]
    host = normalize_client_host(runtime["host"])
    return (
        f"http://{host}:{runtime['frontend_port']}".rstrip("/"),
        f"http://{host}:{runtime['backend_port']}".rstrip("/"),
    )


FRONTEND_URL, BACKEND_URL = resolve_test_urls()


###############################################################################
@pytest.fixture(scope="session")
def base_url() -> str:
    return FRONTEND_URL


###############################################################################
@pytest.fixture(scope="session")
def api_base_url() -> str:
    return BACKEND_URL


###############################################################################
@pytest.fixture(scope="session")
def api_context(playwright: Playwright, api_base_url: str) -> APIRequestContext:
    context = playwright.request.new_context(base_url=api_base_url)
    yield context
    context.dispose()


###############################################################################
@pytest.fixture(scope="session")
def ml_api_base_url(api_base_url: str) -> str:
    """Compatibility fixture for ML-focused tests, using the one canonical backend."""
    return api_base_url


###############################################################################
@pytest.fixture(scope="session")
def ml_api_context(playwright: Playwright, api_base_url: str) -> APIRequestContext:
    context = playwright.request.new_context(base_url=api_base_url)
    capabilities = context.get("/api/v1/system/capabilities")
    if not capabilities.ok or not capabilities.json().get("features", {}).get("machine_learning"):
        context.dispose()
        pytest.skip("Optional machine learning dependencies are not installed.")
    yield context
    context.dispose()


###############################################################################
@pytest.fixture
def page_context(page: Page, base_url: str) -> Page:
    page.goto(base_url)
    return page


###############################################################################
@pytest.fixture(scope="session")
def sample_csv_path() -> Path:
    return FIXTURES_DIR / "sample_adsorption.csv"
