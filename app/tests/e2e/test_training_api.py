"""E2E tests for the training and ML endpoints."""

from __future__ import annotations

import os

import pytest
from playwright.sync_api import APIRequestContext

###############################################################################
class TestTrainingDatasets:
    """Tests for training dataset availability endpoint."""

    # -------------------------------------------------------------------------
    def test_training_datasets_response_shape(
        self, ml_api_context: APIRequestContext
    ) -> None:
        """Verify training dataset availability and conditional metadata."""
        response = ml_api_context.get("/api/v1/training/datasets")

        assert response.ok
        data = response.json()
        assert "available" in data
        if data.get("available"):
            assert "train_samples" in data or "name" in data

###############################################################################
class TestCheckpoints:
    """Tests for the checkpoints listing endpoint."""

    # -------------------------------------------------------------------------
    def test_get_checkpoints(self, ml_api_context: APIRequestContext) -> None:
        """Verify checkpoints endpoint returns a list."""
        response = ml_api_context.get("/api/v1/training/checkpoints")

        assert response.ok
        data = response.json()
        assert "checkpoints" in data
        assert isinstance(data["checkpoints"], list)

###############################################################################
class TestTrainingStatus:
    """Tests for the training status endpoint."""

    # -------------------------------------------------------------------------
    def test_training_status_response_shape(
        self, ml_api_context: APIRequestContext
    ) -> None:
        """Verify training status includes progress information."""
        response = ml_api_context.get("/api/v1/training/status")

        assert response.ok
        data = response.json()
        assert "is_training" in data
        assert "current_epoch" in data
        assert "total_epochs" in data
        assert "progress" in data

###############################################################################
class TestDatasetInfo:
    """Tests for the dataset info endpoint."""

    # -------------------------------------------------------------------------
    def test_get_dataset_info(self, ml_api_context: APIRequestContext) -> None:
        """Verify dataset info endpoint returns expected structure."""
        response = ml_api_context.get("/api/v1/training/dataset-info")

        assert response.ok
        assert "available" in response.json()

###############################################################################
class TestDatasetBuild:
    """Tests for the dataset build endpoint."""

    # -------------------------------------------------------------------------
    def test_build_dataset_request_structure(
        self,
        api_context: APIRequestContext,
        ml_api_context: APIRequestContext,
    ) -> None:
        """Verify dataset build endpoint accepts valid request without server errors."""
        nist_status = api_context.get("/api/v1/nist/status")
        if nist_status.ok:
            status_payload = nist_status.json()
            if status_payload.get("data_available"):
                max_rows = int(os.getenv("TEST_MAX_NIST_ROWS", "1000"))
                row_count = int(status_payload.get("single_component_rows", 0))
                if row_count > max_rows:
                    pytest.skip(
                        f"NIST dataset has {row_count} rows; skip heavy build in tests."
                    )

        payload = {
            "sample_size": float(os.getenv("TEST_DATASET_SAMPLE_SIZE", "0.02")),
            "validation_size": 0.2,
            "min_measurements": 2,
            "max_measurements": 10,
            "smile_sequence_size": 16,
            "max_pressure": 5000.0,
            "max_uptake": 10.0,
            "datasets": [{"source": "nist", "dataset_name": "NIST ISODB"}],
        }

        response = ml_api_context.post("/api/v1/training/build-dataset", data=payload)

        assert response.status in (200, 400)
        if response.ok:
            assert "job_id" in response.json()

    # -------------------------------------------------------------------------
    def test_build_dataset_invalid_params(
        self, ml_api_context: APIRequestContext
    ) -> None:
        """Verify dataset build with invalid params returns a validation error."""
        payload = {
            "sample_size": 2.0,
            "validation_size": 0.2,
            "datasets": [{"source": "nist", "dataset_name": "NIST ISODB"}],
        }

        response = ml_api_context.post("/api/v1/training/build-dataset", data=payload)

        assert response.status == 422

###############################################################################
class TestClearDataset:
    """Tests for the clear dataset endpoint."""

    # -------------------------------------------------------------------------
    def test_clear_training_dataset(self, ml_api_context: APIRequestContext) -> None:
        """Verify clear dataset endpoint responds."""
        response = ml_api_context.delete("/api/v1/training/dataset")

        assert response.ok
        data = response.json()
        assert data.get("status") in {"success", "error"}
        assert "message" in data

###############################################################################
class TestDatasetSources:
    """Tests for the training source catalog."""

    # -------------------------------------------------------------------------
    def test_unregistered_dataset_source_delete_route_returns_not_found(
        self, ml_api_context: APIRequestContext
    ) -> None:
        response = ml_api_context.delete(
            "/api/v1/training/dataset-source?source=uploaded&dataset_name=missing-dataset"
        )

        assert response.status == 404

###############################################################################
class TestTrainingLifecycle:
    """Tests for training start/resume/stop behavior."""

    # -------------------------------------------------------------------------
    def test_start_training_when_dataset_missing(
        self, ml_api_context: APIRequestContext
    ) -> None:
        """Verify start training fails when no dataset is available."""
        dataset_response = ml_api_context.get("/api/v1/training/datasets")
        assert dataset_response.ok
        if dataset_response.json().get("available"):
            pytest.skip("Training dataset exists; avoid starting a real session.")

        response = ml_api_context.post("/api/v1/training/start", data={"epochs": 1})

        assert response.status == 400

    # -------------------------------------------------------------------------
    def test_resume_training_with_missing_checkpoint(
        self, ml_api_context: APIRequestContext
    ) -> None:
        """Verify resume training fails for a missing checkpoint."""
        checkpoints_response = ml_api_context.get("/api/v1/training/checkpoints")
        assert checkpoints_response.ok
        if checkpoints_response.json().get("checkpoints"):
            pytest.skip("Checkpoints exist; avoid resuming a real session.")

        response = ml_api_context.post(
            "/api/v1/training/resume",
            data={"checkpoint_name": "missing-checkpoint", "additional_epochs": 1},
        )

        assert response.status == 404

    # -------------------------------------------------------------------------
    def test_stop_training_when_idle(self, ml_api_context: APIRequestContext) -> None:
        """Verify stop training succeeds when no session is active."""
        response = ml_api_context.post("/api/v1/training/stop")

        assert response.ok
        data = response.json()
        assert data.get("status") == "stopped"
        assert "message" in data
