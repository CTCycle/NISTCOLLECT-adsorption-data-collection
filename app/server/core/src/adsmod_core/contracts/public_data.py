from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SourceStatus = Literal["available", "degraded", "unavailable", "unknown"]
ProviderCapability = Literal[
    "adsorption",
    "materials",
    "chemicals",
    "structures",
    "references",
]


###############################################################################
class PublicSourceSummary(BaseModel):
    key: str
    name: str
    description: str
    capabilities: list[ProviderCapability]
    status: SourceStatus = "unknown"
    status_detail: str | None = None
    homepage_url: str
    license_name: str | None = None
    license_url: str | None = None
    terms_url: str | None = None
    record_count: int = 0
    last_checked_at: datetime | None = None


###############################################################################
class PublicSourceListResponse(BaseModel):
    sources: list[PublicSourceSummary]


###############################################################################
class Pagination(BaseModel):
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)


###############################################################################
class ExternalIdentifierView(BaseModel):
    source: str
    external_id: str
    source_url: str | None = None
    retrieved_at: datetime | None = None
    source_version: str | None = None


###############################################################################
class AdsorptionRecordView(BaseModel):
    id: int
    external_id: str
    source: str
    source_url: str | None = None
    material: str
    adsorbates: list[str]
    temperature_k: float
    pressure_min_pa: float | None = None
    pressure_max_pa: float | None = None
    uptake_min_mol_kg: float | None = None
    uptake_max_mol_kg: float | None = None
    point_count: int = 0
    reference: str | None = None
    retrieved_at: datetime | None = None


###############################################################################
class AdsorptionPageResponse(BaseModel):
    items: list[AdsorptionRecordView]
    pagination: Pagination


###############################################################################
class MeasurementView(BaseModel):
    sequence_index: int
    adsorbate: str
    pressure_original: float
    pressure_original_unit: str
    pressure_pa: float
    uptake_original: float
    uptake_original_unit: str
    uptake_mol_kg: float


###############################################################################
class AdsorptionDetailResponse(AdsorptionRecordView):
    pressure_basis: str
    conditions: dict[str, object]
    provenance: dict[str, object]
    measurements: list[MeasurementView]
    external_identifiers: list[ExternalIdentifierView]


###############################################################################
class MaterialRecordView(BaseModel):
    id: int
    name: str
    formula: str | None = None
    molar_mass_g_mol: float | None = None
    structure_count: int = 0
    external_identifiers: list[ExternalIdentifierView]


###############################################################################
class MaterialPageResponse(BaseModel):
    items: list[MaterialRecordView]
    pagination: Pagination


###############################################################################
class ChemicalPropertyView(BaseModel):
    key: str
    value_number: float | None = None
    value_text: str | None = None
    unit: str | None = None
    source: str


###############################################################################
class ChemicalRecordView(BaseModel):
    id: int
    name: str
    preferred_name: str | None = None
    formula: str | None = None
    molecular_weight: float | None = None
    inchi: str | None = None
    inchi_key: str | None = None
    connectivity_smiles: str | None = None
    smiles: str | None = None
    pubchem_cid: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    properties: list[ChemicalPropertyView] = Field(default_factory=list)
    external_identifiers: list[ExternalIdentifierView] = Field(default_factory=list)
    structure_2d_url: str | None = None
    conformer_3d_url: str | None = None
    retrieved_at: datetime | None = None


###############################################################################
class ChemicalPageResponse(BaseModel):
    items: list[ChemicalRecordView]
    pagination: Pagination


###############################################################################
class PubChemResolveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=512)


###############################################################################
class CODSearchResult(BaseModel):
    cod_id: str
    name: str | None = None
    formula: str | None = None
    space_group: str | None = None
    space_group_number: int | None = None
    cell_a_angstrom: float | None = None
    cell_b_angstrom: float | None = None
    cell_c_angstrom: float | None = None
    cell_alpha_deg: float | None = None
    cell_beta_deg: float | None = None
    cell_gamma_deg: float | None = None
    cell_volume_angstrom3: float | None = None
    doi: str | None = None
    year: int | None = None
    has_coordinates: bool = False
    source_url: str
    cif_url: str


###############################################################################
class CODSearchResponse(BaseModel):
    items: list[CODSearchResult]


###############################################################################
class CODStructureImportRequest(BaseModel):
    cod_id: str = Field(pattern=r"^\d{4,12}$")
    adsorbent_id: int | None = Field(default=None, ge=1)


###############################################################################
class StructureAtomView(BaseModel):
    sequence_index: int
    label: str
    element: str
    fractional_x: float
    fractional_y: float
    fractional_z: float
    occupancy: float | None = None


###############################################################################
class StructureRecordView(BaseModel):
    id: int
    source: str
    external_id: str
    source_url: str | None = None
    material_id: int | None = None
    material_name: str | None = None
    name: str | None = None
    formula: str | None = None
    format: str
    content_sha256: str
    space_group: str | None = None
    space_group_number: int | None = None
    cell_a_angstrom: float | None = None
    cell_b_angstrom: float | None = None
    cell_c_angstrom: float | None = None
    cell_alpha_deg: float | None = None
    cell_beta_deg: float | None = None
    cell_gamma_deg: float | None = None
    cell_volume_angstrom3: float | None = None
    has_coordinates: bool
    atom_count: int
    doi: str | None = None
    retrieved_at: datetime | None = None
    atoms: list[StructureAtomView] = Field(default_factory=list)


###############################################################################
class StructurePageResponse(BaseModel):
    items: list[StructureRecordView]
    pagination: Pagination


__all__ = [
    "AdsorptionDetailResponse",
    "AdsorptionPageResponse",
    "AdsorptionRecordView",
    "ChemicalPageResponse",
    "ChemicalPropertyView",
    "ChemicalRecordView",
    "CODSearchResponse",
    "CODSearchResult",
    "CODStructureImportRequest",
    "ExternalIdentifierView",
    "MaterialPageResponse",
    "MaterialRecordView",
    "MeasurementView",
    "Pagination",
    "ProviderCapability",
    "PublicSourceListResponse",
    "PublicSourceSummary",
    "PubChemResolveRequest",
    "SourceStatus",
    "StructureAtomView",
    "StructurePageResponse",
    "StructureRecordView",
]
