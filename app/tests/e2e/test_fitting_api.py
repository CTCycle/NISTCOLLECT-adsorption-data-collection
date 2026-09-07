"""E2E tests for the fitting pipeline endpoints."""

from __future__ import annotations

import os
import time
import uuid

from playwright.sync_api import APIRequestContext

from .test_datasets_api import _commit_sample

###############################################################################
class TestFittingRun:
    """Tests for the fitting run endpoint."""

    # -------------------------------------------------------------------------
    @staticmethod
    def _max_evaluations(default_value: int) -> int:
        value = os.getenv("TEST_MAX_FITTING_EVALUATIONS")
        if value is None:
            return default_value
        try:
            return max(1, int(value))
        except ValueError:
            return default_value

    # -------------------------------------------------------------------------
    @staticmethod
    def _fitting_target(
        api_context: APIRequestContext, sample_csv_path: str
    ) -> tuple[int, int]:
        dataset = _commit_sample(
            api_context,
            sample_csv_path,
            f"fitting_test_{uuid.uuid4().hex[:8]}",
        )
        response = api_context.get(f"/api/v1/datasets/{dataset['id']}/experiments")
        assert response.ok, response.text()
        experiment = next(
            item for item in response.json()["experiments"] if item["fitting_eligible"]
        )
        return dataset["id"], experiment["id"]

    # -------------------------------------------------------------------------
    @staticmethod
    def _wait_for_job_completion(
        api_context: APIRequestContext,
        job_id: str,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.5,
    ) -> dict:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            status_response = api_context.get(f"/api/v1/fitting/jobs/{job_id}")
            if not status_response.ok:
                raise AssertionError(
                    f"Failed to fetch job status: {status_response.text()}"
                )
            payload = status_response.json()
            status = payload.get("status")
            if status in {"completed", "failed", "cancelled"}:
                return payload
            time.sleep(poll_interval_seconds)
        raise AssertionError(f"Job {job_id} did not complete within timeout.")

    # -------------------------------------------------------------------------
    def test_run_fitting_langmuir(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify fitting with Langmuir model succeeds."""
        dataset_id, isotherm_id = self._fitting_target(api_context, sample_csv_path)

        payload = {
            "dataset_id": dataset_id,
            "isotherm_id": isotherm_id,
            "models": ["langmuir"],
            "optimizer": "trf",
            "max_evaluations": self._max_evaluations(100),
        }

        # Act
        response = api_context.post("/api/v1/fitting/run", data=payload)

        # Assert
        assert response.ok, f"Fitting failed: {response.text()}"
        data = response.json()
        assert "job_id" in data
        job_status = self._wait_for_job_completion(api_context, data["job_id"])
        if job_status.get("status") != "completed":
            raise AssertionError(f"Job did not complete successfully: {job_status}")

    # -------------------------------------------------------------------------
    def test_run_fitting_multiple_models(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify fitting with multiple models succeeds."""
        dataset_id, isotherm_id = self._fitting_target(api_context, sample_csv_path)

        payload = {
            "dataset_id": dataset_id,
            "isotherm_id": isotherm_id,
            "models": ["langmuir", "freundlich"],
            "optimizer": "trf",
            "max_evaluations": self._max_evaluations(120),
        }

        # Act
        response = api_context.post("/api/v1/fitting/run", data=payload)

        # Assert
        assert response.ok
        data = response.json()
        assert "job_id" in data
        job_status = self._wait_for_job_completion(api_context, data["job_id"])
        if job_status.get("status") != "completed":
            raise AssertionError(f"Job did not complete successfully: {job_status}")

    # -------------------------------------------------------------------------
    def test_run_fitting_invalid_optimizer(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        """Verify an unsupported optimizer is rejected by request validation."""
        dataset_id, isotherm_id = self._fitting_target(api_context, sample_csv_path)

        payload = {
            "dataset_id": dataset_id,
            "isotherm_id": isotherm_id,
            "models": ["langmuir"],
            "optimizer": "INVALID_METHOD",
        }

        # Act
        response = api_context.post("/api/v1/fitting/run", data=payload)

        # Assert
        assert response.status == 422  # Pydantic validation error

###############################################################################
class TestModelCatalog:
    """Tests for the current fitting model catalog endpoint."""

    # -------------------------------------------------------------------------
    def test_get_model_catalog(self, api_context: APIRequestContext) -> None:
        response = api_context.get("/api/v1/fitting/models")

        assert response.ok, response.text()
        data = response.json()
        assert data["status"] == "success"
        assert data["pressure_unit"] == "bar"
        assert data["uptake_unit"] == "mmol/g"
        models = {model["key"]: model for model in data["models"]}
        assert {"langmuir", "freundlich"}.issubset(models)
        assert {
            parameter["name"] for parameter in models["langmuir"]["parameters"]
        } == {"k", "qsat"}

###############################################################################
class TestFittingJobs:
    """Tests for fitting job polling and cancellation payloads."""

    # -------------------------------------------------------------------------
    def test_cancel_unknown_job_returns_error(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.delete("/api/v1/fitting/jobs/unknown-job")
        assert response.status == 400
