"""E2E tests for the canonical dataset import and management endpoints."""

from __future__ import annotations

import json
import uuid
from typing import Any

from playwright.sync_api import APIRequestContext

###############################################################################
def _read_sample(sample_csv_path: str) -> bytes:
    with open(sample_csv_path, "rb") as handle:
        return handle.read()

###############################################################################
def _build_mapping(
    api_context: APIRequestContext,
    sample_csv_path: str,
    dataset_name: str,
) -> tuple[bytes, dict[str, Any]]:
    file_content = _read_sample(sample_csv_path)
    response = api_context.post(
        "/api/v1/datasets/import/preview",
        multipart={
            "file": {
                "name": f"{dataset_name}.csv",
                "mimeType": "text/csv",
                "buffer": file_content,
            }
        },
    )
    assert response.ok, f"Preview failed: {response.text()}"
    preview = response.json()
    column_roles = {
        column["name"]: column["proposed_role"] for column in preview["columns"]
    }
    grouping_columns = preview.get("proposed_grouping_columns") or [
        name
        for name, role in column_roles.items()
        if role in {"experiment_id", "experiment_name"}
    ]
    unit_overrides = {
        column["proposed_role"]: column["detected_unit"]
        for column in preview["columns"]
        if column.get("detected_unit")
        and column["proposed_role"] in {"pressure", "uptake", "temperature"}
    }
    mapping: dict[str, Any] = {
        "dataset_name": dataset_name,
        "structure": preview["detected_structure"],
        "column_roles": column_roles,
        "grouping_columns": grouping_columns,
        "pressure_basis": preview.get("proposed_pressure_basis") or "absolute",
        "unit_overrides": unit_overrides,
        "decimal_separator": ".",
        "duplicate_policy": "keep",
    }
    return file_content, mapping

###############################################################################
def _commit_sample(
    api_context: APIRequestContext,
    sample_csv_path: str,
    dataset_name: str,
) -> dict[str, Any]:
    file_content, mapping = _build_mapping(api_context, sample_csv_path, dataset_name)
    validation_response = api_context.post(
        "/api/v1/datasets/import/validate",
        multipart={
            "mapping": json.dumps(mapping),
            "file": {
                "name": f"{dataset_name}.csv",
                "mimeType": "text/csv",
                "buffer": file_content,
            },
        },
    )
    assert validation_response.ok, f"Validation failed: {validation_response.text()}"
    assert validation_response.json()["status"] == "valid"

    commit_response = api_context.post(
        "/api/v1/datasets/import/commit",
        multipart={
            "mapping": json.dumps(mapping),
            "file": {
                "name": f"{dataset_name}.csv",
                "mimeType": "text/csv",
                "buffer": file_content,
            },
        },
    )
    assert commit_response.ok, f"Commit failed: {commit_response.text()}"
    return commit_response.json()["dataset"]

###############################################################################
class TestDatasetImport:
    """Tests for the four-stage canonical dataset import flow."""

    # -------------------------------------------------------------------------
    def test_import_csv_dataset(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        dataset_name = f"test_adsorption_{uuid.uuid4().hex[:8]}"
        dataset = _commit_sample(api_context, sample_csv_path, dataset_name)

        assert dataset["name"] == dataset_name
        assert dataset["source"] == "uploaded"
        assert dataset["experiment_count"] > 0
        assert dataset["observation_count"] > 0

###############################################################################
class TestDatasetList:
    """Tests for listing canonical dataset summaries."""

    # -------------------------------------------------------------------------
    def test_get_dataset_list(self, api_context: APIRequestContext) -> None:
        response = api_context.get("/api/v1/datasets")

        assert response.ok, response.text()
        data = response.json()
        assert data["status"] == "success"
        assert isinstance(data["datasets"], list)

###############################################################################
class TestDatasetExperiments:
    """Tests for fetching experiments from an imported dataset."""

    # -------------------------------------------------------------------------
    def test_get_experiments_after_import(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        dataset = _commit_sample(
            api_context,
            sample_csv_path,
            f"experiment_test_{uuid.uuid4().hex[:8]}",
        )
        response = api_context.get(f"/api/v1/datasets/{dataset['id']}/experiments")

        assert response.ok, response.text()
        data = response.json()
        assert data["status"] == "success"
        assert len(data["experiments"]) > 0
        assert data["experiments"][0]["dataset_id"] == dataset["id"]

    # -------------------------------------------------------------------------
    def test_get_nonexistent_dataset_experiments(
        self, api_context: APIRequestContext
    ) -> None:
        response = api_context.get("/api/v1/datasets/999999/experiments")

        assert response.status == 404

###############################################################################
class TestDatasetDeletion:
    """Tests for deletion through the canonical dataset API."""

    # -------------------------------------------------------------------------
    def test_delete_uploaded_dataset_by_canonical_id(
        self, api_context: APIRequestContext, sample_csv_path: str
    ) -> None:
        dataset = _commit_sample(
            api_context,
            sample_csv_path,
            f"delete_test_{uuid.uuid4().hex[:8]}",
        )

        response = api_context.delete(f"/api/v1/datasets/{dataset['id']}")

        assert response.status == 204
        listing = api_context.get("/api/v1/datasets")
        assert listing.ok
        assert dataset["id"] not in {item["id"] for item in listing.json()["datasets"]}
