from __future__ import annotations

import asyncio
from typing import Any

from adsmod_core.contracts.public_data import (
    AdsorptionDetailResponse,
    AdsorptionPageResponse,
    CODSearchResponse,
    CODStructureImportRequest,
    ChemicalPageResponse,
    ChemicalRecordView,
    MaterialPageResponse,
    Pagination,
    PublicSourceListResponse,
    PublicSourceSummary,
    StructurePageResponse,
    StructureRecordView,
)
from adsmod_core.providers.cod import CODProvider
from adsmod_core.providers.pubchem import PubChemProvider
from adsmod_core.providers.public_data import PublicDataProvider
from adsmod_core.repositories.public_data import PublicDataRepository


###############################################################################
class PublicDataService:
    """Coordinate normalized public-data queries and provider adapters."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        repository: PublicDataRepository,
        providers: list[PublicDataProvider],
    ) -> None:
        self.repository = repository
        self.providers = {provider.key: provider for provider in providers}

    # -------------------------------------------------------------------------
    async def close(self) -> None:
        await asyncio.gather(
            *(provider.close() for provider in self.providers.values()),
            return_exceptions=True,
        )

    # -------------------------------------------------------------------------
    async def list_sources(self, *, check_health: bool = True) -> PublicSourceListResponse:
        persisted = {row["key"]: row for row in self.repository.source_rows()}
        health_by_key: dict[str, Any] = {}
        if check_health:
            keys = list(self.providers)
            health = await asyncio.gather(
                *(self.providers[key].health() for key in keys),
                return_exceptions=True,
            )
            for key, result in zip(keys, health, strict=True):
                if isinstance(result, BaseException):
                    health_by_key[key] = {
                        "status": "unavailable",
                        "detail": str(result),
                        "checked_at": None,
                    }
                else:
                    health_by_key[key] = {
                        "status": result.status,
                        "detail": result.detail,
                        "checked_at": result.checked_at,
                    }

        items: list[PublicSourceSummary] = []
        for key, provider in sorted(self.providers.items(), key=lambda item: item[1].name):
            row = persisted.get(key, {})
            health = health_by_key.get(key, {})
            items.append(
                PublicSourceSummary(
                    key=key,
                    name=provider.name,
                    description=provider.description,
                    capabilities=[capability.value for capability in provider.capabilities],
                    status=health.get("status", "unknown"),
                    status_detail=health.get("detail"),
                    homepage_url=provider.homepage_url,
                    license_name=provider.license_name,
                    license_url=provider.license_url,
                    terms_url=provider.terms_url,
                    record_count=int(row.get("record_count", 0)),
                    last_checked_at=health.get("checked_at"),
                )
            )
        return PublicSourceListResponse(sources=items)

    # -------------------------------------------------------------------------
    def list_adsorption(self, **filters: Any) -> AdsorptionPageResponse:
        rows, total = self.repository.list_adsorption(**filters)
        return AdsorptionPageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def get_adsorption(self, isotherm_id: int) -> AdsorptionDetailResponse:
        return AdsorptionDetailResponse.model_validate(
            self.repository.get_adsorption(isotherm_id)
        )

    # -------------------------------------------------------------------------
    def list_materials(self, **filters: Any) -> MaterialPageResponse:
        rows, total = self.repository.list_materials(**filters)
        return MaterialPageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def list_chemicals(self, **filters: Any) -> ChemicalPageResponse:
        rows, total = self.repository.list_chemicals(**filters)
        return ChemicalPageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def get_chemical(self, adsorbate_id: int) -> ChemicalRecordView:
        return ChemicalRecordView.model_validate(
            self.repository.get_chemical(adsorbate_id)
        )

    # -------------------------------------------------------------------------
    async def resolve_pubchem(self, query: str) -> ChemicalRecordView:
        provider = self.providers.get("pubchem")
        if not isinstance(provider, PubChemProvider):
            raise RuntimeError("PubChem provider is not configured.")
        payload = await provider.resolve(query)
        adsorbate_id = await asyncio.to_thread(
            self.repository.upsert_pubchem_compound, payload
        )
        return self.get_chemical(adsorbate_id)

    # -------------------------------------------------------------------------
    async def search_cod(
        self,
        *,
        text: str | None,
        formula: str | None,
        cod_id: str | None,
    ) -> CODSearchResponse:
        provider = self.providers.get("cod")
        if not isinstance(provider, CODProvider):
            raise RuntimeError("COD provider is not configured.")
        rows = await provider.search(text=text, formula=formula, cod_id=cod_id)
        return CODSearchResponse(items=rows)

    # -------------------------------------------------------------------------
    async def import_cod(
        self, request: CODStructureImportRequest
    ) -> StructureRecordView:
        provider = self.providers.get("cod")
        if not isinstance(provider, CODProvider):
            raise RuntimeError("COD provider is not configured.")
        metadata, cif_text = await provider.fetch_record(request.cod_id)
        atoms = provider.parse_atoms(cif_text)
        structure_id = await asyncio.to_thread(
            self.repository.upsert_cod_structure,
            metadata=metadata,
            cif_text=cif_text,
            atoms=atoms,
            adsorbent_id=request.adsorbent_id,
        )
        return StructureRecordView.model_validate(
            self.repository.get_structure(structure_id)
        )

    # -------------------------------------------------------------------------
    def list_structures(self, **filters: Any) -> StructurePageResponse:
        rows, total = self.repository.list_structures(**filters)
        return StructurePageResponse(
            items=rows,
            pagination=Pagination(
                page=int(filters["page"]),
                page_size=int(filters["page_size"]),
                total=total,
            ),
        )

    # -------------------------------------------------------------------------
    def get_structure(self, structure_id: int) -> StructureRecordView:
        return StructureRecordView.model_validate(
            self.repository.get_structure(structure_id)
        )


__all__ = ["PublicDataService"]
