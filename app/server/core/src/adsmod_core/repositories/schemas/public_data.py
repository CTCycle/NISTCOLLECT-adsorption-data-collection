from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from adsmod_core.repositories.schemas.models import Base, utc_now
from adsmod_core.repositories.schemas.types import JSONList, JSONMapping, UTCDateTime


###############################################################################
class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    homepage_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    license_name: Mapped[str | None] = mapped_column(String(128))
    license_url: Mapped[str | None] = mapped_column(String(1024))
    terms_url: Mapped[str | None] = mapped_column(String(1024))
    capabilities: Mapped[list[Any]] = mapped_column(JSONList, nullable=False, default=list)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False
    )


###############################################################################
class SourceRecord(Base):
    __tablename__ = "source_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("data_sources.id", ondelete="RESTRICT"), nullable=False
    )
    record_type: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str] = mapped_column(String(512), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2048))
    source_version: Mapped[str | None] = mapped_column(String(128))
    transform_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="public-data-v1"
    )
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONMapping, nullable=False, default=dict
    )

    __table_args__ = (
        UniqueConstraint(
            "source_id", "record_type", "external_id", name="uq_source_records_identity"
        ),
        CheckConstraint(
            "record_type IN ('adsorption', 'chemical', 'material', 'structure', 'reference')",
            name="ck_source_records_type",
        ),
        Index("ix_source_records_source_retrieved", "source_id", "retrieved_at"),
    )


###############################################################################
class AdsorbateSourceRecord(Base):
    __tablename__ = "adsorbate_source_records"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    adsorbate_id: Mapped[int] = mapped_column(
        ForeignKey("adsorbates.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "adsorbate_id", "source_record_id", name="uq_adsorbate_source_record"
        ),
        Index("ix_adsorbate_source_records_adsorbate", "adsorbate_id"),
    )


###############################################################################
class AdsorbentSourceRecord(Base):
    __tablename__ = "adsorbent_source_records"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    adsorbent_id: Mapped[int] = mapped_column(
        ForeignKey("adsorbents.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "adsorbent_id", "source_record_id", name="uq_adsorbent_source_record"
        ),
        Index("ix_adsorbent_source_records_adsorbent", "adsorbent_id"),
    )


###############################################################################
class IsothermSourceRecord(Base):
    __tablename__ = "isotherm_source_records"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    isotherm_id: Mapped[int] = mapped_column(
        ForeignKey("isotherms.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "isotherm_id", "source_record_id", name="uq_isotherm_source_record"
        ),
        Index("ix_isotherm_source_records_isotherm", "isotherm_id"),
    )


###############################################################################
class Structure(Base):
    __tablename__ = "structures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adsorbent_id: Mapped[int | None] = mapped_column(
        ForeignKey("adsorbents.id", ondelete="SET NULL")
    )
    name: Mapped[str | None] = mapped_column(String(512))
    formula: Mapped[str | None] = mapped_column(String(512))
    format: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    space_group: Mapped[str | None] = mapped_column(String(128))
    space_group_number: Mapped[int | None] = mapped_column(Integer)
    cell_a_angstrom: Mapped[float | None] = mapped_column(Float)
    cell_b_angstrom: Mapped[float | None] = mapped_column(Float)
    cell_c_angstrom: Mapped[float | None] = mapped_column(Float)
    cell_alpha_deg: Mapped[float | None] = mapped_column(Float)
    cell_beta_deg: Mapped[float | None] = mapped_column(Float)
    cell_gamma_deg: Mapped[float | None] = mapped_column(Float)
    cell_volume_angstrom3: Mapped[float | None] = mapped_column(Float)
    has_coordinates: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime, default=utc_now, onupdate=utc_now, nullable=False
    )

    __table_args__ = (
        CheckConstraint("format IN ('cif', 'sdf', 'xyz')", name="ck_structures_format"),
        Index("ix_structures_adsorbent", "adsorbent_id"),
        Index("ix_structures_formula", "formula"),
    )


###############################################################################
class StructureSourceRecord(Base):
    __tablename__ = "structure_source_records"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    structure_id: Mapped[int] = mapped_column(
        ForeignKey("structures.id", ondelete="CASCADE"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "structure_id", "source_record_id", name="uq_structure_source_record"
        ),
        Index("ix_structure_source_records_structure", "structure_id"),
    )


###############################################################################
class StructureAtom(Base):
    __tablename__ = "structure_atoms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    structure_id: Mapped[int] = mapped_column(
        ForeignKey("structures.id", ondelete="CASCADE"), nullable=False
    )
    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    element: Mapped[str] = mapped_column(String(8), nullable=False)
    fractional_x: Mapped[float] = mapped_column(Float, nullable=False)
    fractional_y: Mapped[float] = mapped_column(Float, nullable=False)
    fractional_z: Mapped[float] = mapped_column(Float, nullable=False)
    occupancy: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint(
            "structure_id", "sequence_index", name="uq_structure_atoms_sequence"
        ),
        CheckConstraint("sequence_index >= 0", name="ck_structure_atoms_sequence"),
        CheckConstraint(
            "occupancy IS NULL OR (occupancy >= 0 AND occupancy <= 1)",
            name="ck_structure_atoms_occupancy",
        ),
        Index("ix_structure_atoms_structure", "structure_id", "sequence_index"),
    )


###############################################################################
class AdsorbateSynonym(Base):
    __tablename__ = "adsorbate_synonyms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adsorbate_id: Mapped[int] = mapped_column(
        ForeignKey("adsorbates.id", ondelete="CASCADE"), nullable=False
    )
    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False
    )
    synonym: Mapped[str] = mapped_column(String(1024), nullable=False)
    normalized_synonym: Mapped[str] = mapped_column(String(1024), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "adsorbate_id",
            "source_record_id",
            "normalized_synonym",
            name="uq_adsorbate_synonyms_identity",
        ),
        Index("ix_adsorbate_synonyms_normalized", "normalized_synonym"),
    )


###############################################################################
class ChemicalProperty(Base):
    __tablename__ = "chemical_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adsorbate_id: Mapped[int] = mapped_column(
        ForeignKey("adsorbates.id", ondelete="CASCADE"), nullable=False
    )
    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value_number: Mapped[float | None] = mapped_column(Float)
    value_text: Mapped[str | None] = mapped_column(String(2048))
    unit: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint(
            "adsorbate_id", "source_record_id", "key", name="uq_chemical_properties_key"
        ),
        CheckConstraint(
            "value_number IS NOT NULL OR value_text IS NOT NULL",
            name="ck_chemical_properties_value",
        ),
        Index("ix_chemical_properties_adsorbate_key", "adsorbate_id", "key"),
    )


###############################################################################
class MaterialProperty(Base):
    __tablename__ = "material_properties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adsorbent_id: Mapped[int] = mapped_column(
        ForeignKey("adsorbents.id", ondelete="CASCADE"), nullable=False
    )
    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False
    )
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    value_number: Mapped[float | None] = mapped_column(Float)
    value_text: Mapped[str | None] = mapped_column(String(2048))
    unit: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint(
            "adsorbent_id", "source_record_id", "key", name="uq_material_properties_key"
        ),
        CheckConstraint(
            "value_number IS NOT NULL OR value_text IS NOT NULL",
            name="ck_material_properties_value",
        ),
        Index("ix_material_properties_adsorbent_key", "adsorbent_id", "key"),
    )


###############################################################################
class Reference(Base):
    __tablename__ = "references"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doi: Mapped[str | None] = mapped_column(String(255), unique=True)
    title: Mapped[str | None] = mapped_column(String(2048))
    journal: Mapped[str | None] = mapped_column(String(512))
    year: Mapped[int | None] = mapped_column(Integer)
    url: Mapped[str | None] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utc_now, nullable=False)

    __table_args__ = (Index("ix_references_year", "year"),)


###############################################################################
class SourceRecordReference(Base):
    __tablename__ = "source_record_references"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    reference_id: Mapped[int] = mapped_column(
        ForeignKey("references.id", ondelete="CASCADE"), primary_key=True
    )


__all__ = [
    "AdsorbateSourceRecord",
    "AdsorbateSynonym",
    "AdsorbentSourceRecord",
    "ChemicalProperty",
    "DataSource",
    "IsothermSourceRecord",
    "MaterialProperty",
    "Reference",
    "SourceRecord",
    "SourceRecordReference",
    "Structure",
    "StructureAtom",
    "StructureSourceRecord",
]
