from __future__ import annotations

import json
import hashlib
from collections.abc import Collection
from pathlib import Path

from pydantic import ValidationError

from adsmod_common.config import DEFAULT_DATASET_ALLOWED_EXTENSIONS
from adsmod_core.contracts.datasets import (
    DatasetImportResponse,
    DatasetListResponse,
    DatasetMetadata,
    DatasetMutationResponse,
    DatasetSummary,
    ExperimentListResponse,
    ImportMapping,
    ImportPreviewResponse,
    ImportValidationResponse,
    ObservationPage,
    SupportedUnitsResponse,
)
from adsmod_core.services.data.importer import AdsorptionImportEngine, PARSER_VERSION
from adsmod_common.units import UnitRegistry
from adsmod_core.repositories.datasets import DatasetRepository

###############################################################################
class DatasetService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        repository: DatasetRepository,
        importer: AdsorptionImportEngine | None = None,
        allowed_extensions: Collection[str] = DEFAULT_DATASET_ALLOWED_EXTENSIONS,
    ) -> None:
        self.repository = repository
        self.importer = importer or AdsorptionImportEngine(allowed_extensions)
        self.allowed_extensions = list(
            getattr(self.importer, "allowed_extensions", allowed_extensions)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def parse_mapping(mapping_json: str) -> ImportMapping:
        try:
            decoded = json.loads(mapping_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Import mapping must be valid JSON.") from exc
        try:
            return ImportMapping.model_validate(decoded)
        except ValidationError as exc:
            first = exc.errors()[0]
            raise ValueError(str(first.get("msg", "Invalid import mapping."))) from exc

    # -------------------------------------------------------------------------
    def preview(self, payload: bytes, filename: str | None) -> ImportPreviewResponse:
        return self.importer.preview(payload, filename)

    # -------------------------------------------------------------------------
    def validate(
        self, payload: bytes, filename: str | None, mapping: ImportMapping
    ) -> ImportValidationResponse:
        return self.importer.validate(payload, filename, mapping).response

    # -------------------------------------------------------------------------
    def commit(
        self, payload: bytes, filename: str | None, mapping: ImportMapping
    ) -> DatasetImportResponse:
        bundle = self.importer.validate(payload, filename, mapping)
        if bundle.response.status != "valid":
            raise ValueError(
                "Dataset validation must succeed and all required confirmations "
                "must be acknowledged before saving."
            )
        manifest = {
            "original_filename": Path(filename or "dataset").name,
            "source_sha256": bundle.response.source_sha256,
            "source_structure": mapping.structure,
            "parser_version": PARSER_VERSION,
            "column_mapping": mapping.model_dump(mode="json"),
            "validation_result": bundle.response.model_dump(mode="json"),
            "warnings": [
                issue.model_dump(mode="json")
                for issue in bundle.response.issues
                if issue.severity == "warning"
            ],
        }
        mapping_sha256 = hashlib.sha256(
            json.dumps(
                mapping.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        manifest["mapping_sha256"] = mapping_sha256
        dataset_id = self.repository.persist_canonical(
            name=mapping.dataset_name,
            source="uploaded",
            provenance={
                "origin": "user_upload",
                "original_filename": Path(filename or "dataset").name,
                "source_sha256": bundle.response.source_sha256,
                "mapping_sha256": mapping_sha256,
            },
            experiments=bundle.experiments,
            import_manifest=manifest,
        )
        return DatasetImportResponse(
            dataset=DatasetSummary(**self.repository.summary(dataset_id)),
            validation=bundle.response,
        )

    # -------------------------------------------------------------------------
    def list_datasets(self) -> DatasetListResponse:
        return DatasetListResponse(
            datasets=[
                DatasetSummary(**record) for record in self.repository.list_summaries()
            ]
        )

    # -------------------------------------------------------------------------
    def supported_units(self) -> SupportedUnitsResponse:
        return SupportedUnitsResponse(
            pressure=sorted(UnitRegistry.PRESSURE_ALIASES),
            uptake=sorted(UnitRegistry.UPTAKE_ALIASES),
            temperature=sorted(UnitRegistry.TEMPERATURE_ALIASES),
            allowed_extensions=self.allowed_extensions,
        )

    # -------------------------------------------------------------------------
    def list_experiments(self, dataset_id: int) -> ExperimentListResponse:
        return ExperimentListResponse(
            experiments=self.repository.experiments(dataset_id)
        )

    # -------------------------------------------------------------------------
    def get_observations(
        self, dataset_id: int, isotherm_id: int, offset: int, limit: int
    ) -> ObservationPage:
        rows, total = self.repository.observations(
            dataset_id, isotherm_id, offset, limit
        )
        return ObservationPage(
            dataset_id=dataset_id,
            isotherm_id=isotherm_id,
            offset=offset,
            limit=limit,
            total=total,
            rows=rows,
        )

    # -------------------------------------------------------------------------
    def rename(self, dataset_id: int, new_name: str) -> DatasetMutationResponse:
        self.repository.rename(dataset_id, new_name)
        return DatasetMutationResponse(
            dataset=DatasetSummary(**self.repository.summary(dataset_id))
        )

    # -------------------------------------------------------------------------
    def update_metadata(
        self, dataset_id: int, metadata: DatasetMetadata
    ) -> DatasetMutationResponse:
        self.repository.update_metadata(dataset_id, metadata.tags, metadata.description)
        return DatasetMutationResponse(
            dataset=DatasetSummary(**self.repository.summary(dataset_id))
        )

    # -------------------------------------------------------------------------
    def delete(self, dataset_id: int) -> None:
        self.repository.delete(dataset_id)
