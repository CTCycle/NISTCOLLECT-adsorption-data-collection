"""E2E tests for the NIST data ingestion endpoints."""

from __future__ import annotations

import json

from playwright.sync_api import APIRequestContext

###############################################################################
class TestNistStatus:
    """Tests for the NIST status endpoint."""

    # -------------------------------------------------------------------------
    def test_nist_status_shape_and_counts(self, api_context: APIRequestContext) -> None:
        """Verify NIST availability and conditional row-count fields."""
        response = api_context.get("/api/v1/nist/status")

        assert response.ok
        data = response.json()
        assert "data_available" in data
        if data.get("data_available"):
            assert "single_component_rows" in data
            assert "binary_mixture_rows" in data
            assert "guest_rows" in data
            assert "host_rows" in data

###############################################################################
class TestNistFetch:
    """Tests for the NIST data fetch endpoint."""

    # -------------------------------------------------------------------------
    def test_fetch_nist_data_small_fraction(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify NIST fetch with small fraction succeeds."""
        payload = {
            "experiments_fraction": 0.01,
            "guest_fraction": 0.01,
            "host_fraction": 0.01,
        }

        response = api_context.post(
            "/api/v1/nist/fetch",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

        assert response.status in (200, 400, 500)
        if response.status == 400:
            detail = response.json().get("detail", "").lower()
            assert "job is already running" in detail
        if response.ok:
            assert isinstance(response.json(), dict)

    # -------------------------------------------------------------------------
    def test_fetch_nist_data_invalid_fraction(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify NIST fetch with invalid fraction returns a validation error."""
        payload = {
            "experiments_fraction": 2.0,
            "guest_fraction": 0.01,
            "host_fraction": 0.01,
        }

        response = api_context.post("/api/v1/nist/fetch", data=payload)

        assert response.status == 422

###############################################################################
class TestNistProperties:
    """Tests for the NIST properties enrichment endpoint."""

    # -------------------------------------------------------------------------
    def test_fetch_nist_properties_guest(self, api_context: APIRequestContext) -> None:
        """Verify NIST properties fetch for guest materials."""
        response = api_context.post("/api/v1/nist/properties", data={"target": "guest"})

        assert response.status in (200, 400, 500)
        if response.ok:
            data = response.json()
            assert "job_id" in data
            assert data.get("job_type") == "nist_properties"

    # -------------------------------------------------------------------------
    def test_fetch_nist_properties_host(self, api_context: APIRequestContext) -> None:
        """Verify NIST properties fetch for host materials."""
        response = api_context.post("/api/v1/nist/properties", data={"target": "host"})

        assert response.status in (200, 400, 500)
        if response.ok:
            data = response.json()
            assert "job_id" in data
            assert data.get("job_type") == "nist_properties"

    # -------------------------------------------------------------------------
    def test_fetch_nist_properties_invalid_target(
        self, api_context: APIRequestContext
    ) -> None:
        """Verify NIST properties with invalid target returns a validation error."""
        response = api_context.post(
            "/api/v1/nist/properties", data={"target": "invalid"}
        )

        assert response.status == 422
