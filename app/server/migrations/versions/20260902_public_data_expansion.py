"""Add normalized multi-source public data and structural provenance.

Revision ID: 20260902_public_data
Revises: 20260829_v3
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from alembic import op
import sqlalchemy as sa

from adsmod_core.repositories.schemas import types as schema_types


revision: str = "20260902_public_data"
down_revision: str | None = "20260829_v3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


###############################################################################
def _now() -> datetime:
    return datetime.now(timezone.utc)


###############################################################################
def upgrade() -> None:
    op.create_table(
        "data_sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("homepage_url", sa.String(length=1024), nullable=False),
        sa.Column("license_name", sa.String(length=128), nullable=True),
        sa.Column("license_url", sa.String(length=1024), nullable=True),
        sa.Column("terms_url", sa.String(length=1024), nullable=True),
        sa.Column("capabilities", schema_types.JSONList(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", schema_types.UTCDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", schema_types.UTCDateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_table(
        "source_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("record_type", sa.String(length=32), nullable=False),
        sa.Column("external_id", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("source_version", sa.String(length=128), nullable=True),
        sa.Column("transform_version", sa.String(length=64), nullable=False),
        sa.Column("retrieved_at", schema_types.UTCDateTime(timezone=True), nullable=False),
        sa.Column("raw_metadata", schema_types.JSONMapping(), nullable=False),
        sa.CheckConstraint(
            "record_type IN ('adsorption', 'chemical', 'material', 'structure', 'reference')",
            name="ck_source_records_type",
        ),
        sa.ForeignKeyConstraint(["source_id"], ["data_sources.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_id", "record_type", "external_id", name="uq_source_records_identity"
        ),
    )
    op.create_index(
        "ix_source_records_source_retrieved",
        "source_records",
        ["source_id", "retrieved_at"],
        unique=False,
    )

    op.create_table(
        "adsorbate_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["adsorbate_id"], ["adsorbates.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "adsorbate_id", "source_record_id", name="uq_adsorbate_source_record"
        ),
    )
    op.create_index(
        "ix_adsorbate_source_records_adsorbate",
        "adsorbate_source_records",
        ["adsorbate_id"],
        unique=False,
    )
    op.create_table(
        "adsorbent_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("adsorbent_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["adsorbent_id"], ["adsorbents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "adsorbent_id", "source_record_id", name="uq_adsorbent_source_record"
        ),
    )
    op.create_index(
        "ix_adsorbent_source_records_adsorbent",
        "adsorbent_source_records",
        ["adsorbent_id"],
        unique=False,
    )
    op.create_table(
        "isotherm_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("isotherm_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["isotherm_id"], ["isotherms.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "isotherm_id", "source_record_id", name="uq_isotherm_source_record"
        ),
    )
    op.create_index(
        "ix_isotherm_source_records_isotherm",
        "isotherm_source_records",
        ["isotherm_id"],
        unique=False,
    )

    op.create_table(
        "structures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbent_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=512), nullable=True),
        sa.Column("formula", sa.String(length=512), nullable=True),
        sa.Column("format", sa.String(length=16), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("space_group", sa.String(length=128), nullable=True),
        sa.Column("space_group_number", sa.Integer(), nullable=True),
        sa.Column("cell_a_angstrom", sa.Float(), nullable=True),
        sa.Column("cell_b_angstrom", sa.Float(), nullable=True),
        sa.Column("cell_c_angstrom", sa.Float(), nullable=True),
        sa.Column("cell_alpha_deg", sa.Float(), nullable=True),
        sa.Column("cell_beta_deg", sa.Float(), nullable=True),
        sa.Column("cell_gamma_deg", sa.Float(), nullable=True),
        sa.Column("cell_volume_angstrom3", sa.Float(), nullable=True),
        sa.Column("has_coordinates", sa.Boolean(), nullable=False),
        sa.Column("created_at", schema_types.UTCDateTime(timezone=True), nullable=False),
        sa.Column("updated_at", schema_types.UTCDateTime(timezone=True), nullable=False),
        sa.CheckConstraint("format IN ('cif', 'sdf', 'xyz')", name="ck_structures_format"),
        sa.ForeignKeyConstraint(["adsorbent_id"], ["adsorbents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_structures_adsorbent", "structures", ["adsorbent_id"], unique=False)
    op.create_index("ix_structures_formula", "structures", ["formula"], unique=False)
    op.create_table(
        "structure_source_records",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("structure_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["structure_id"], ["structures.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_record_id"),
        sa.UniqueConstraint(
            "structure_id", "source_record_id", name="uq_structure_source_record"
        ),
    )
    op.create_index(
        "ix_structure_source_records_structure",
        "structure_source_records",
        ["structure_id"],
        unique=False,
    )
    op.create_table(
        "structure_atoms",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("structure_id", sa.Integer(), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("element", sa.String(length=8), nullable=False),
        sa.Column("fractional_x", sa.Float(), nullable=False),
        sa.Column("fractional_y", sa.Float(), nullable=False),
        sa.Column("fractional_z", sa.Float(), nullable=False),
        sa.Column("occupancy", sa.Float(), nullable=True),
        sa.CheckConstraint("sequence_index >= 0", name="ck_structure_atoms_sequence"),
        sa.CheckConstraint(
            "occupancy IS NULL OR (occupancy >= 0 AND occupancy <= 1)",
            name="ck_structure_atoms_occupancy",
        ),
        sa.ForeignKeyConstraint(["structure_id"], ["structures.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "structure_id", "sequence_index", name="uq_structure_atoms_sequence"
        ),
    )
    op.create_index(
        "ix_structure_atoms_structure",
        "structure_atoms",
        ["structure_id", "sequence_index"],
        unique=False,
    )

    op.create_table(
        "adsorbate_synonyms",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("synonym", sa.String(length=1024), nullable=False),
        sa.Column("normalized_synonym", sa.String(length=1024), nullable=False),
        sa.ForeignKeyConstraint(["adsorbate_id"], ["adsorbates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "adsorbate_id",
            "source_record_id",
            "normalized_synonym",
            name="uq_adsorbate_synonyms_identity",
        ),
    )
    op.create_index(
        "ix_adsorbate_synonyms_normalized",
        "adsorbate_synonyms",
        ["normalized_synonym"],
        unique=False,
    )
    op.create_table(
        "chemical_properties",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbate_id", sa.Integer(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value_number", sa.Float(), nullable=True),
        sa.Column("value_text", sa.String(length=2048), nullable=True),
        sa.Column("unit", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "value_number IS NOT NULL OR value_text IS NOT NULL",
            name="ck_chemical_properties_value",
        ),
        sa.ForeignKeyConstraint(["adsorbate_id"], ["adsorbates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "adsorbate_id", "source_record_id", "key", name="uq_chemical_properties_key"
        ),
    )
    op.create_index(
        "ix_chemical_properties_adsorbate_key",
        "chemical_properties",
        ["adsorbate_id", "key"],
        unique=False,
    )
    op.create_table(
        "material_properties",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("adsorbent_id", sa.Integer(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=128), nullable=False),
        sa.Column("value_number", sa.Float(), nullable=True),
        sa.Column("value_text", sa.String(length=2048), nullable=True),
        sa.Column("unit", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "value_number IS NOT NULL OR value_text IS NOT NULL",
            name="ck_material_properties_value",
        ),
        sa.ForeignKeyConstraint(["adsorbent_id"], ["adsorbents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "adsorbent_id", "source_record_id", "key", name="uq_material_properties_key"
        ),
    )
    op.create_index(
        "ix_material_properties_adsorbent_key",
        "material_properties",
        ["adsorbent_id", "key"],
        unique=False,
    )

    op.create_table(
        "references",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("doi", sa.String(length=255), nullable=True),
        sa.Column("title", sa.String(length=2048), nullable=True),
        sa.Column("journal", sa.String(length=512), nullable=True),
        sa.Column("year", sa.Integer(), nullable=True),
        sa.Column("url", sa.String(length=2048), nullable=True),
        sa.Column("created_at", schema_types.UTCDateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("doi"),
    )
    op.create_index("ix_references_year", "references", ["year"], unique=False)
    op.create_table(
        "source_record_references",
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("reference_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["reference_id"], ["references.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_record_id"], ["source_records.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("source_record_id", "reference_id"),
    )

    source_table = sa.table(
        "data_sources",
        sa.column("key", sa.String()),
        sa.column("name", sa.String()),
        sa.column("description", sa.String()),
        sa.column("homepage_url", sa.String()),
        sa.column("license_name", sa.String()),
        sa.column("license_url", sa.String()),
        sa.column("terms_url", sa.String()),
        sa.column("capabilities", schema_types.JSONList()),
        sa.column("enabled", sa.Boolean()),
        sa.column("created_at", schema_types.UTCDateTime(timezone=True)),
        sa.column("updated_at", schema_types.UTCDateTime(timezone=True)),
    )
    now = _now()
    op.bulk_insert(
        source_table,
        [
            {
                "key": "nist",
                "name": "NIST/ARPA-E Adsorption Database",
                "description": "Adsorption experiments, guest species, and host materials.",
                "homepage_url": "https://adsorption.nist.gov/",
                "license_name": None,
                "license_url": None,
                "terms_url": "https://adsorption.nist.gov/",
                "capabilities": ["adsorption", "materials", "chemicals", "references"],
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            },
            {
                "key": "pubchem",
                "name": "PubChem",
                "description": "Chemical identities, descriptors, synonyms, and molecular structures.",
                "homepage_url": "https://pubchem.ncbi.nlm.nih.gov/",
                "license_name": None,
                "license_url": "https://pubchem.ncbi.nlm.nih.gov/docs/downloads",
                "terms_url": "https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access",
                "capabilities": ["chemicals", "structures", "references"],
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            },
            {
                "key": "cod",
                "name": "Crystallography Open Database",
                "description": "Open crystal structures, CIF records, and publication metadata.",
                "homepage_url": "https://www.crystallography.net/cod/",
                "license_name": "CC0 1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "terms_url": "https://wiki.crystallography.net/howtoobtaincod/",
                "capabilities": ["materials", "structures", "references"],
                "enabled": True,
                "created_at": now,
                "updated_at": now,
            },
        ],
    )

    _backfill_nist_provenance()


###############################################################################
def _backfill_nist_provenance() -> None:
    bind = op.get_bind()
    source_id = bind.execute(
        sa.text("SELECT id FROM data_sources WHERE key = :key"), {"key": "nist"}
    ).scalar_one()
    now = _now()

    isotherms = bind.execute(
        sa.text(
            """
            SELECT i.id, i.external_key
            FROM isotherms AS i
            JOIN datasets AS d ON d.id = i.dataset_id
            WHERE d.source = 'nist'
            """
        )
    ).all()
    for isotherm_id, external_key in isotherms:
        record_id = _source_record_id(
            bind,
            source_id=source_id,
            record_type="adsorption",
            external_id=str(external_key),
            source_url="https://adsorption.nist.gov/",
            now=now,
        )
        bind.execute(
            sa.text(
                "INSERT INTO isotherm_source_records (source_record_id, isotherm_id) "
                "VALUES (:source_record_id, :isotherm_id)"
            ),
            {"source_record_id": record_id, "isotherm_id": isotherm_id},
        )

    adsorbates = bind.execute(
        sa.text(
            """
            SELECT DISTINCT a.id, a.inchi_key, a.key
            FROM adsorbates AS a
            JOIN isotherm_components AS c ON c.adsorbate_id = a.id
            JOIN isotherms AS i ON i.id = c.isotherm_id
            JOIN datasets AS d ON d.id = i.dataset_id
            WHERE d.source = 'nist'
            """
        )
    ).all()
    for adsorbate_id, inchi_key, fallback_key in adsorbates:
        external_id = str(inchi_key or fallback_key)
        record_id = _source_record_id(
            bind,
            source_id=source_id,
            record_type="chemical",
            external_id=external_id,
            source_url=(
                f"https://adsorption.nist.gov/isodb/api/gas/{external_id}.json"
                if inchi_key
                else "https://adsorption.nist.gov/"
            ),
            now=now,
        )
        bind.execute(
            sa.text(
                "INSERT INTO adsorbate_source_records (source_record_id, adsorbate_id) "
                "VALUES (:source_record_id, :adsorbate_id)"
            ),
            {"source_record_id": record_id, "adsorbate_id": adsorbate_id},
        )

    adsorbents = bind.execute(
        sa.text(
            """
            SELECT DISTINCT a.id, a.external_identifier, a.key
            FROM adsorbents AS a
            JOIN isotherms AS i ON i.adsorbent_id = a.id
            JOIN datasets AS d ON d.id = i.dataset_id
            WHERE d.source = 'nist'
            """
        )
    ).all()
    for adsorbent_id, external_identifier, fallback_key in adsorbents:
        external_id = str(external_identifier or fallback_key)
        record_id = _source_record_id(
            bind,
            source_id=source_id,
            record_type="material",
            external_id=external_id,
            source_url=(
                f"https://adsorption.nist.gov/matdb/api/material/{external_id}.json"
                if external_identifier
                else "https://adsorption.nist.gov/"
            ),
            now=now,
        )
        bind.execute(
            sa.text(
                "INSERT INTO adsorbent_source_records (source_record_id, adsorbent_id) "
                "VALUES (:source_record_id, :adsorbent_id)"
            ),
            {"source_record_id": record_id, "adsorbent_id": adsorbent_id},
        )


###############################################################################
def _source_record_id(
    bind: sa.Connection,
    *,
    source_id: int,
    record_type: str,
    external_id: str,
    source_url: str,
    now: datetime,
) -> int:
    bind.execute(
        sa.text(
            """
            INSERT INTO source_records (
                source_id, record_type, external_id, source_url, source_version,
                transform_version, retrieved_at, raw_metadata
            ) VALUES (
                :source_id, :record_type, :external_id, :source_url, NULL,
                :transform_version, :retrieved_at, :raw_metadata
            )
            """
        ),
        {
            "source_id": source_id,
            "record_type": record_type,
            "external_id": external_id,
            "source_url": source_url,
            "transform_version": "public-data-v1-backfill",
            "retrieved_at": now,
            "raw_metadata": "{}",
        },
    )
    return int(
        bind.execute(
            sa.text(
                "SELECT id FROM source_records WHERE source_id = :source_id "
                "AND record_type = :record_type AND external_id = :external_id"
            ),
            {
                "source_id": source_id,
                "record_type": record_type,
                "external_id": external_id,
            },
        ).scalar_one()
    )


###############################################################################
def downgrade() -> None:
    op.drop_table("source_record_references")
    op.drop_index("ix_references_year", table_name="references")
    op.drop_table("references")
    op.drop_index("ix_material_properties_adsorbent_key", table_name="material_properties")
    op.drop_table("material_properties")
    op.drop_index("ix_chemical_properties_adsorbate_key", table_name="chemical_properties")
    op.drop_table("chemical_properties")
    op.drop_index("ix_adsorbate_synonyms_normalized", table_name="adsorbate_synonyms")
    op.drop_table("adsorbate_synonyms")
    op.drop_index("ix_structure_atoms_structure", table_name="structure_atoms")
    op.drop_table("structure_atoms")
    op.drop_index("ix_structure_source_records_structure", table_name="structure_source_records")
    op.drop_table("structure_source_records")
    op.drop_index("ix_structures_formula", table_name="structures")
    op.drop_index("ix_structures_adsorbent", table_name="structures")
    op.drop_table("structures")
    op.drop_index("ix_isotherm_source_records_isotherm", table_name="isotherm_source_records")
    op.drop_table("isotherm_source_records")
    op.drop_index("ix_adsorbent_source_records_adsorbent", table_name="adsorbent_source_records")
    op.drop_table("adsorbent_source_records")
    op.drop_index("ix_adsorbate_source_records_adsorbate", table_name="adsorbate_source_records")
    op.drop_table("adsorbate_source_records")
    op.drop_index("ix_source_records_source_retrieved", table_name="source_records")
    op.drop_table("source_records")
    op.drop_table("data_sources")
