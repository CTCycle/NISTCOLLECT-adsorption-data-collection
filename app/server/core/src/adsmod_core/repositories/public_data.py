from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.datasets import stable_material_key
from adsmod_core.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)
from adsmod_core.repositories.schemas.public_data import (
    AdsorbateSourceRecord,
    AdsorbateSynonym,
    AdsorbentSourceRecord,
    ChemicalProperty,
    DataSource,
    IsothermSourceRecord,
    Reference,
    SourceRecord,
    SourceRecordReference,
    Structure,
    StructureAtom,
    StructureSourceRecord,
)
from adsmod_core.repositories.schemas.types import normalize_identity


SOURCE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "nist",
        "name": "NIST/ARPA-E Adsorption Database",
        "description": "Adsorption experiments, guest species, and host materials.",
        "homepage_url": "https://adsorption.nist.gov/",
        "license_name": None,
        "license_url": None,
        "terms_url": "https://adsorption.nist.gov/",
        "capabilities": ["adsorption", "materials", "chemicals", "references"],
    },
    {
        "key": "pubchem",
        "name": "PubChem",
        "description": "Chemical identities, descriptors, synonyms, and molecular structures.",
        "homepage_url": "https://pubchem.ncbi.nlm.nih.gov/",
        "license_name": "Public domain / source-attributed records",
        "license_url": "https://pubchem.ncbi.nlm.nih.gov/docs/downloads",
        "terms_url": "https://pubchem.ncbi.nlm.nih.gov/docs/programmatic-access",
        "capabilities": ["chemicals", "structures", "references"],
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
    },
)


###############################################################################
class PublicDataRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def ensure_sources(self) -> None:
        with self.database.transaction() as session:
            for definition in SOURCE_DEFINITIONS:
                source = session.scalar(
                    select(DataSource).where(DataSource.key == definition["key"])
                )
                if source is None:
                    session.add(DataSource(**definition))
                    continue
                for key, value in definition.items():
                    if key != "key":
                        setattr(source, key, value)
                source.enabled = True

    # -------------------------------------------------------------------------
    @staticmethod
    def _source(session: Session, key: str) -> DataSource:
        source = session.scalar(select(DataSource).where(DataSource.key == key))
        if source is None:
            raise LookupError(f"Public data source {key!r} is not registered.")
        return source

    # -------------------------------------------------------------------------
    @staticmethod
    def _record(
        session: Session,
        *,
        source: DataSource,
        record_type: str,
        external_id: str,
        source_url: str | None = None,
        source_version: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ) -> SourceRecord:
        record = session.scalar(
            select(SourceRecord).where(
                SourceRecord.source_id == source.id,
                SourceRecord.record_type == record_type,
                SourceRecord.external_id == external_id,
            )
        )
        if record is None:
            record = SourceRecord(
                source_id=source.id,
                record_type=record_type,
                external_id=external_id,
                source_url=source_url,
                source_version=source_version,
                retrieved_at=datetime.now(timezone.utc),
                raw_metadata=raw_metadata or {},
            )
            session.add(record)
            session.flush()
        else:
            record.source_url = source_url or record.source_url
            record.source_version = source_version or record.source_version
            record.retrieved_at = datetime.now(timezone.utc)
            if raw_metadata is not None:
                record.raw_metadata = raw_metadata
        return record

    # -------------------------------------------------------------------------
    def source_rows(self) -> list[dict[str, Any]]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(DataSource, func.count(SourceRecord.id))
                .outerjoin(SourceRecord, SourceRecord.source_id == DataSource.id)
                .where(DataSource.enabled.is_(True))
                .group_by(DataSource.id)
                .order_by(DataSource.name)
            ).all()
            return [
                {
                    "key": source.key,
                    "name": source.name,
                    "description": source.description,
                    "homepage_url": source.homepage_url,
                    "license_name": source.license_name,
                    "license_url": source.license_url,
                    "terms_url": source.terms_url,
                    "capabilities": list(source.capabilities),
                    "record_count": int(count or 0),
                }
                for source, count in rows
            ]

    # -------------------------------------------------------------------------
    def link_adsorbate_record(
        self,
        *,
        source_key: str,
        adsorbate_id: int,
        external_id: str,
        source_url: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.database.transaction() as session:
            source = self._source(session, source_key)
            record = self._record(
                session,
                source=source,
                record_type="chemical",
                external_id=external_id,
                source_url=source_url,
                raw_metadata=raw_metadata,
            )
            link = session.get(AdsorbateSourceRecord, record.id)
            if link is not None and link.adsorbate_id != adsorbate_id:
                raise ValueError(
                    f"{source_key}:{external_id} is already linked to another adsorbate."
                )
            if link is None:
                session.add(
                    AdsorbateSourceRecord(
                        source_record_id=record.id, adsorbate_id=adsorbate_id
                    )
                )

    # -------------------------------------------------------------------------
    def link_adsorbent_record(
        self,
        *,
        source_key: str,
        adsorbent_id: int,
        external_id: str,
        source_url: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.database.transaction() as session:
            source = self._source(session, source_key)
            record = self._record(
                session,
                source=source,
                record_type="material",
                external_id=external_id,
                source_url=source_url,
                raw_metadata=raw_metadata,
            )
            link = session.get(AdsorbentSourceRecord, record.id)
            if link is not None and link.adsorbent_id != adsorbent_id:
                raise ValueError(
                    f"{source_key}:{external_id} is already linked to another material."
                )
            if link is None:
                session.add(
                    AdsorbentSourceRecord(
                        source_record_id=record.id, adsorbent_id=adsorbent_id
                    )
                )

    # -------------------------------------------------------------------------
    def link_isotherm_record(
        self,
        *,
        source_key: str,
        dataset_id: int,
        external_id: str,
        source_url: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.database.transaction() as session:
            isotherm_id = session.scalar(
                select(Isotherm.id).where(
                    Isotherm.dataset_id == dataset_id,
                    Isotherm.external_key == external_id,
                )
            )
            if isotherm_id is None:
                raise LookupError(f"Isotherm {external_id!r} was not persisted.")
            source = self._source(session, source_key)
            record = self._record(
                session,
                source=source,
                record_type="adsorption",
                external_id=external_id,
                source_url=source_url,
                raw_metadata=raw_metadata,
            )
            link = session.get(IsothermSourceRecord, record.id)
            if link is not None and link.isotherm_id != isotherm_id:
                raise ValueError(
                    f"{source_key}:{external_id} is already linked to another isotherm."
                )
            if link is None:
                session.add(
                    IsothermSourceRecord(
                        source_record_id=record.id, isotherm_id=isotherm_id
                    )
                )

    # -------------------------------------------------------------------------
    def upsert_pubchem_compound(self, payload: dict[str, Any]) -> int:
        cid = str(payload["cid"])
        inchi_key = str(payload.get("inchi_key") or "").strip() or None
        name = str(payload.get("name") or payload.get("preferred_name") or cid).strip()
        with self.database.transaction() as session:
            source = self._source(session, "pubchem")
            record = self._record(
                session,
                source=source,
                record_type="chemical",
                external_id=cid,
                source_url=payload.get("source_url"),
                raw_metadata=payload.get("raw_metadata") or {},
            )
            existing_link = session.get(AdsorbateSourceRecord, record.id)
            adsorbate: Adsorbate | None = None
            if existing_link is not None:
                adsorbate = session.get(Adsorbate, existing_link.adsorbate_id)
            elif inchi_key:
                adsorbate = session.scalar(
                    select(Adsorbate).where(Adsorbate.inchi_key == inchi_key)
                )
            if adsorbate is None:
                adsorbate = Adsorbate(
                    key=stable_material_key("adsorbate", name, inchi_key or f"pubchem:{cid}"),
                    name=name,
                    inchi_key=inchi_key,
                )
                session.add(adsorbate)
                session.flush()
            elif inchi_key and adsorbate.inchi_key not in (None, inchi_key):
                raise ValueError(
                    "PubChem identity conflicts with the canonical adsorbate InChIKey; "
                    "automatic merging was refused."
                )

            adsorbate.name = name or adsorbate.name
            adsorbate.normalized_name = normalize_identity(adsorbate.name)
            adsorbate.inchi_key = inchi_key or adsorbate.inchi_key
            adsorbate.inchi = payload.get("inchi") or adsorbate.inchi
            adsorbate.formula = payload.get("formula") or adsorbate.formula
            adsorbate.molar_mass_g_mol = (
                payload.get("molecular_weight")
                if payload.get("molecular_weight") is not None
                else adsorbate.molar_mass_g_mol
            )
            adsorbate.smiles = payload.get("smiles") or adsorbate.smiles
            if existing_link is None:
                session.add(
                    AdsorbateSourceRecord(
                        source_record_id=record.id, adsorbate_id=adsorbate.id
                    )
                )

            session.execute(
                delete(AdsorbateSynonym).where(
                    AdsorbateSynonym.source_record_id == record.id
                )
            )
            seen: set[str] = set()
            for synonym in payload.get("synonyms") or []:
                value = str(synonym).strip()
                normalized = normalize_identity(value) if value else ""
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                session.add(
                    AdsorbateSynonym(
                        adsorbate_id=adsorbate.id,
                        source_record_id=record.id,
                        synonym=value,
                        normalized_synonym=normalized,
                    )
                )

            session.execute(
                delete(ChemicalProperty).where(
                    ChemicalProperty.source_record_id == record.id
                )
            )
            property_rows: list[tuple[str, float | None, str | None, str | None]] = []
            preferred_name = payload.get("preferred_name")
            if preferred_name:
                property_rows.append(("preferred_name", None, str(preferred_name), None))
            connectivity_smiles = payload.get("connectivity_smiles")
            if connectivity_smiles:
                property_rows.append(
                    ("connectivity_smiles", None, str(connectivity_smiles), None)
                )
            if payload.get("conformer_3d_url"):
                property_rows.append(("pubchem_3d_available", None, "true", None))
            units = {
                "tpsa_angstrom2": "Å²",
                "exact_mass_da": "Da",
                "monoisotopic_mass_da": "Da",
            }
            for key, value in (payload.get("descriptors") or {}).items():
                property_rows.append((str(key), float(value), None, units.get(str(key))))
            for element, value in (payload.get("elemental_composition") or {}).items():
                property_rows.append((f"element:{element}", float(value), None, "atoms"))
            for key, value_number, value_text, unit in property_rows:
                session.add(
                    ChemicalProperty(
                        adsorbate_id=adsorbate.id,
                        source_record_id=record.id,
                        key=key,
                        value_number=value_number,
                        value_text=value_text,
                        unit=unit,
                    )
                )
            session.flush()
            return adsorbate.id

    # -------------------------------------------------------------------------
    def upsert_cod_structure(
        self,
        *,
        metadata: dict[str, Any],
        cif_text: str,
        atoms: list[dict[str, Any]],
        adsorbent_id: int | None,
    ) -> int:
        cod_id = str(metadata["cod_id"])
        with self.database.transaction() as session:
            if adsorbent_id is not None and session.get(Adsorbent, adsorbent_id) is None:
                raise LookupError(f"Material {adsorbent_id} does not exist.")
            source = self._source(session, "cod")
            record = self._record(
                session,
                source=source,
                record_type="structure",
                external_id=cod_id,
                source_url=metadata.get("source_url"),
                source_version=metadata.get("source_version"),
                raw_metadata=metadata.get("raw_metadata") or {},
            )
            link = session.get(StructureSourceRecord, record.id)
            structure = session.get(Structure, link.structure_id) if link else None
            if structure is None:
                structure = Structure(
                    format="cif",
                    content=cif_text,
                    content_sha256=hashlib.sha256(cif_text.encode("utf-8")).hexdigest(),
                )
                session.add(structure)
                session.flush()
                session.add(
                    StructureSourceRecord(
                        source_record_id=record.id, structure_id=structure.id
                    )
                )
            if adsorbent_id is not None:
                structure.adsorbent_id = adsorbent_id
            structure.name = metadata.get("name")
            structure.formula = metadata.get("formula")
            structure.content = cif_text
            structure.content_sha256 = hashlib.sha256(cif_text.encode("utf-8")).hexdigest()
            structure.space_group = metadata.get("space_group")
            structure.space_group_number = metadata.get("space_group_number")
            structure.cell_a_angstrom = metadata.get("cell_a_angstrom")
            structure.cell_b_angstrom = metadata.get("cell_b_angstrom")
            structure.cell_c_angstrom = metadata.get("cell_c_angstrom")
            structure.cell_alpha_deg = metadata.get("cell_alpha_deg")
            structure.cell_beta_deg = metadata.get("cell_beta_deg")
            structure.cell_gamma_deg = metadata.get("cell_gamma_deg")
            structure.cell_volume_angstrom3 = metadata.get("cell_volume_angstrom3")
            structure.has_coordinates = bool(atoms) or bool(metadata.get("has_coordinates"))

            session.execute(
                delete(StructureAtom).where(StructureAtom.structure_id == structure.id)
            )
            for atom in atoms:
                session.add(StructureAtom(structure_id=structure.id, **atom))

            doi = str(metadata.get("doi") or "").strip() or None
            if doi:
                reference = session.scalar(
                    select(Reference).where(func.lower(Reference.doi) == doi.lower())
                )
                if reference is None:
                    reference = Reference(
                        doi=doi,
                        title=metadata.get("title"),
                        journal=metadata.get("journal"),
                        year=metadata.get("year"),
                        url=f"https://doi.org/{doi}",
                    )
                    session.add(reference)
                    session.flush()
                link_exists = session.get(
                    SourceRecordReference, (record.id, reference.id)
                )
                if link_exists is None:
                    session.add(
                        SourceRecordReference(
                            source_record_id=record.id, reference_id=reference.id
                        )
                    )
            session.flush()
            return structure.id

    # -------------------------------------------------------------------------
    @staticmethod
    def _provenance_entries(
        session: Session, entity: str, entity_ids: list[int]
    ) -> dict[int, list[tuple[int, dict[str, Any]]]]:
        if not entity_ids:
            return {}
        if entity == "adsorbate":
            link_model = AdsorbateSourceRecord
            id_column = link_model.adsorbate_id
        elif entity == "adsorbent":
            link_model = AdsorbentSourceRecord
            id_column = link_model.adsorbent_id
        elif entity == "isotherm":
            link_model = IsothermSourceRecord
            id_column = link_model.isotherm_id
        elif entity == "structure":
            link_model = StructureSourceRecord
            id_column = link_model.structure_id
        else:
            raise ValueError(f"Unsupported provenance entity: {entity}")

        rows = session.execute(
            select(
                id_column.label("entity_id"),
                SourceRecord.id.label("source_record_id"),
                DataSource.key.label("source_key"),
                SourceRecord.external_id,
                SourceRecord.source_url,
                SourceRecord.retrieved_at,
                SourceRecord.source_version,
            )
            .join(SourceRecord, SourceRecord.source_id == DataSource.id)
            .join(link_model, link_model.source_record_id == SourceRecord.id)
            .where(id_column.in_(entity_ids))
            .order_by(
                id_column,
                SourceRecord.retrieved_at.desc(),
                SourceRecord.id.desc(),
            )
        ).all()
        result: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for row in rows:
            result.setdefault(int(row.entity_id), []).append(
                (
                    int(row.source_record_id),
                    {
                        "source": row.source_key,
                        "external_id": row.external_id,
                        "source_url": row.source_url,
                        "retrieved_at": row.retrieved_at,
                        "source_version": row.source_version,
                    },
                )
            )
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _external_identifiers(
        cls, session: Session, entity: str, entity_id: int
    ) -> list[dict[str, Any]]:
        return [
            identifier
            for _, identifier in cls._provenance_entries(session, entity, [entity_id]).get(
                entity_id, []
            )
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def _references_for_records(
        session: Session, source_record_ids: list[int]
    ) -> dict[int, str | None]:
        if not source_record_ids:
            return {}
        rows = session.execute(
            select(
                SourceRecordReference.source_record_id,
                Reference.id,
                Reference.doi,
                Reference.title,
                Reference.url,
            )
            .join(
                Reference,
                SourceRecordReference.reference_id == Reference.id,
            )
            .where(SourceRecordReference.source_record_id.in_(source_record_ids))
            .order_by(SourceRecordReference.source_record_id, Reference.id)
        ).all()
        result: dict[int, str | None] = {}
        for row in rows:
            result.setdefault(
                int(row.source_record_id), row.doi or row.title or row.url
            )
        return result

    # -------------------------------------------------------------------------
    @classmethod
    def _reference_for_record(
        cls, session: Session, source_record_id: int | None
    ) -> str | None:
        if source_record_id is None:
            return None
        return cls._references_for_records(session, [source_record_id]).get(
            source_record_id
        )

    # -------------------------------------------------------------------------
    def list_adsorption(
        self,
        *,
        page: int,
        page_size: int,
        source: str | None = None,
        material: str | None = None,
        adsorbate: str | None = None,
        temperature_min_k: float | None = None,
        temperature_max_k: float | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = (
                select(
                    Isotherm.id,
                    Isotherm.external_key,
                    Isotherm.temperature_k,
                    Isotherm.pressure_basis,
                    Dataset.source.label("dataset_source"),
                    Adsorbent.name.label("material"),
                )
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
            )
            if source:
                source_ids = (
                    select(IsothermSourceRecord.isotherm_id)
                    .join(
                        SourceRecord,
                        SourceRecord.id == IsothermSourceRecord.source_record_id,
                    )
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(DataSource.key == source)
                )
                query = query.where(Isotherm.id.in_(source_ids))
            if material:
                query = query.where(Adsorbent.name.ilike(f"%{material.strip()}%"))
            if adsorbate:
                matching_adsorbates = (
                    select(IsothermComponent.isotherm_id)
                    .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                    .where(Adsorbate.name.ilike(f"%{adsorbate.strip()}%"))
                )
                query = query.where(Isotherm.id.in_(matching_adsorbates))
            if temperature_min_k is not None:
                query = query.where(Isotherm.temperature_k >= temperature_min_k)
            if temperature_max_k is not None:
                query = query.where(Isotherm.temperature_k <= temperature_max_k)

            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.execute(
                query.order_by(Isotherm.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._adsorption_summaries(session, rows), total

    # -------------------------------------------------------------------------
    def _adsorption_summaries(
        self, session: Session, rows: list[Any]
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        isotherm_ids = [int(row.id) for row in rows]
        adsorbates_by_isotherm: dict[int, list[str]] = {}
        for isotherm_id, adsorbate_name in session.execute(
            select(IsothermComponent.isotherm_id, Adsorbate.name)
            .join(Adsorbate, IsothermComponent.adsorbate_id == Adsorbate.id)
            .where(IsothermComponent.isotherm_id.in_(isotherm_ids))
            .order_by(IsothermComponent.isotherm_id, IsothermComponent.position)
        ):
            adsorbates_by_isotherm.setdefault(int(isotherm_id), []).append(
                str(adsorbate_name)
            )

        stats_by_isotherm = {
            int(row.isotherm_id): row
            for row in session.execute(
                select(
                    Observation.isotherm_id.label("isotherm_id"),
                    func.min(Observation.pressure_canonical).label("pressure_min"),
                    func.max(Observation.pressure_canonical).label("pressure_max"),
                    func.min(Observation.uptake_mol_kg).label("uptake_min"),
                    func.max(Observation.uptake_mol_kg).label("uptake_max"),
                    func.count(Observation.id).label("point_count"),
                )
                .where(Observation.isotherm_id.in_(isotherm_ids))
                .group_by(Observation.isotherm_id)
            )
        }
        provenance_by_isotherm = self._provenance_entries(
            session, "isotherm", isotherm_ids
        )
        primary_record_ids = [
            entries[0][0]
            for entries in provenance_by_isotherm.values()
            if entries
        ]
        references = self._references_for_records(session, primary_record_ids)

        items: list[dict[str, Any]] = []
        for row in rows:
            isotherm_id = int(row.id)
            entries = provenance_by_isotherm.get(isotherm_id, [])
            source_record_id, primary = entries[0] if entries else (None, None)
            stats = stats_by_isotherm.get(isotherm_id)
            items.append(
                {
                    "id": isotherm_id,
                    "external_id": primary["external_id"] if primary else row.external_key,
                    "source": primary["source"] if primary else row.dataset_source,
                    "source_url": primary["source_url"] if primary else None,
                    "material": row.material,
                    "adsorbates": adsorbates_by_isotherm.get(isotherm_id, []),
                    "temperature_k": row.temperature_k,
                    "pressure_min_pa": stats.pressure_min if stats else None,
                    "pressure_max_pa": stats.pressure_max if stats else None,
                    "uptake_min_mol_kg": stats.uptake_min if stats else None,
                    "uptake_max_mol_kg": stats.uptake_max if stats else None,
                    "point_count": int(stats.point_count if stats else 0),
                    "reference": references.get(source_record_id) if source_record_id else None,
                    "retrieved_at": primary["retrieved_at"] if primary else None,
                }
            )
        return items

    # -------------------------------------------------------------------------
    def _adsorption_summary(self, session: Session, row: Any) -> dict[str, Any]:
        return self._adsorption_summaries(session, [row])[0]

    # -------------------------------------------------------------------------
    def get_adsorption(self, isotherm_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            row = session.execute(
                select(
                    Isotherm.id,
                    Isotherm.external_key,
                    Isotherm.temperature_k,
                    Isotherm.pressure_basis,
                    Isotherm.conditions,
                    Isotherm.provenance,
                    Dataset.source.label("dataset_source"),
                    Adsorbent.name.label("material"),
                )
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
                .where(Isotherm.id == isotherm_id)
            ).first()
            if row is None:
                raise LookupError(f"Adsorption record {isotherm_id} was not found.")
            summary = self._adsorption_summary(session, row)
            measurements = session.execute(
                select(
                    Observation.sequence_index,
                    Adsorbate.name,
                    Observation.pressure_original,
                    Observation.pressure_original_unit,
                    Observation.pressure_canonical,
                    Observation.uptake_original,
                    Observation.uptake_original_unit,
                    Observation.uptake_mol_kg,
                )
                .join(
                    IsothermComponent,
                    IsothermComponent.id == Observation.component_id,
                )
                .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                .where(Observation.isotherm_id == isotherm_id)
                .order_by(Observation.sequence_index, IsothermComponent.position)
            ).all()
            summary.update(
                {
                    "pressure_basis": row.pressure_basis,
                    "conditions": dict(row.conditions),
                    "provenance": dict(row.provenance),
                    "measurements": [
                        {
                            "sequence_index": item[0],
                            "adsorbate": item[1],
                            "pressure_original": item[2],
                            "pressure_original_unit": item[3],
                            "pressure_pa": item[4],
                            "uptake_original": item[5],
                            "uptake_original_unit": item[6],
                            "uptake_mol_kg": item[7],
                        }
                        for item in measurements
                    ],
                    "external_identifiers": self._external_identifiers(
                        session, "isotherm", isotherm_id
                    ),
                }
            )
            return summary

    # -------------------------------------------------------------------------
    def list_materials(
        self,
        *,
        page: int,
        page_size: int,
        query_text: str | None = None,
        formula: str | None = None,
        source: str | None = None,
        has_structure: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(Adsorbent)
            if query_text:
                query = query.where(Adsorbent.name.ilike(f"%{query_text.strip()}%"))
            if formula:
                query = query.where(Adsorbent.formula.ilike(f"%{formula.strip()}%"))
            if source:
                ids = (
                    select(AdsorbentSourceRecord.adsorbent_id)
                    .join(
                        SourceRecord,
                        SourceRecord.id == AdsorbentSourceRecord.source_record_id,
                    )
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(DataSource.key == source)
                )
                query = query.where(Adsorbent.id.in_(ids))
            if has_structure is not None:
                structured = select(Structure.adsorbent_id).where(
                    Structure.adsorbent_id.is_not(None)
                )
                query = query.where(
                    Adsorbent.id.in_(structured)
                    if has_structure
                    else Adsorbent.id.not_in(structured)
                )
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(Adsorbent.name)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            row_ids = [item.id for item in rows]
            structure_counts = {
                int(adsorbent_id): int(count)
                for adsorbent_id, count in session.execute(
                    select(Structure.adsorbent_id, func.count(Structure.id))
                    .where(Structure.adsorbent_id.in_(row_ids))
                    .group_by(Structure.adsorbent_id)
                )
                if adsorbent_id is not None
            }
            provenance = self._provenance_entries(session, "adsorbent", row_ids)
            return [
                {
                    "id": item.id,
                    "name": item.name,
                    "formula": item.formula,
                    "molar_mass_g_mol": item.molar_mass_g_mol,
                    "structure_count": structure_counts.get(item.id, 0),
                    "external_identifiers": [
                        identifier
                        for _, identifier in provenance.get(item.id, [])
                    ],
                }
                for item in rows
            ], total

    # -------------------------------------------------------------------------
    def list_chemicals(
        self,
        *,
        page: int,
        page_size: int,
        query_text: str | None = None,
        formula: str | None = None,
        source: str | None = None,
        molecular_weight_min: float | None = None,
        molecular_weight_max: float | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(Adsorbate)
            if query_text:
                normalized = normalize_identity(query_text)
                synonym_ids = select(AdsorbateSynonym.adsorbate_id).where(
                    AdsorbateSynonym.normalized_synonym.ilike(f"%{normalized}%")
                )
                query = query.where(
                    or_(
                        Adsorbate.name.ilike(f"%{query_text.strip()}%"),
                        Adsorbate.id.in_(synonym_ids),
                    )
                )
            if formula:
                query = query.where(Adsorbate.formula.ilike(f"%{formula.strip()}%"))
            if molecular_weight_min is not None:
                query = query.where(Adsorbate.molar_mass_g_mol >= molecular_weight_min)
            if molecular_weight_max is not None:
                query = query.where(Adsorbate.molar_mass_g_mol <= molecular_weight_max)
            if source:
                ids = (
                    select(AdsorbateSourceRecord.adsorbate_id)
                    .join(
                        SourceRecord,
                        SourceRecord.id == AdsorbateSourceRecord.source_record_id,
                    )
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(DataSource.key == source)
                )
                query = query.where(Adsorbate.id.in_(ids))
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(Adsorbate.name)
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._chemical_views(session, rows), total

    # -------------------------------------------------------------------------
    def get_chemical(self, adsorbate_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            item = session.get(Adsorbate, adsorbate_id)
            if item is None:
                raise LookupError(f"Chemical {adsorbate_id} was not found.")
            return self._chemical_view(session, item)

    # -------------------------------------------------------------------------
    def _chemical_views(
        self, session: Session, items: list[Adsorbate]
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        item_ids = [item.id for item in items]
        provenance = self._provenance_entries(session, "adsorbate", item_ids)
        properties_by_adsorbate: dict[int, list[tuple[ChemicalProperty, str]]] = {}
        for adsorbate_id, prop, source_key in session.execute(
            select(ChemicalProperty.adsorbate_id, ChemicalProperty, DataSource.key)
            .join(SourceRecord, SourceRecord.id == ChemicalProperty.source_record_id)
            .join(DataSource, DataSource.id == SourceRecord.source_id)
            .where(ChemicalProperty.adsorbate_id.in_(item_ids))
            .order_by(ChemicalProperty.adsorbate_id, ChemicalProperty.key, ChemicalProperty.id)
        ):
            properties_by_adsorbate.setdefault(int(adsorbate_id), []).append(
                (prop, str(source_key))
            )

        ranked_synonyms = (
            select(
                AdsorbateSynonym.adsorbate_id.label("adsorbate_id"),
                AdsorbateSynonym.synonym.label("synonym"),
                func.row_number()
                .over(
                    partition_by=AdsorbateSynonym.adsorbate_id,
                    order_by=AdsorbateSynonym.synonym,
                )
                .label("row_number"),
            )
            .where(AdsorbateSynonym.adsorbate_id.in_(item_ids))
            .subquery()
        )
        synonyms_by_adsorbate: dict[int, list[str]] = {}
        for adsorbate_id, synonym in session.execute(
            select(ranked_synonyms.c.adsorbate_id, ranked_synonyms.c.synonym)
            .where(ranked_synonyms.c.row_number <= 100)
            .order_by(ranked_synonyms.c.adsorbate_id, ranked_synonyms.c.synonym)
        ):
            synonyms_by_adsorbate.setdefault(int(adsorbate_id), []).append(str(synonym))

        return [
            self._chemical_payload(
                item,
                [identifier for _, identifier in provenance.get(item.id, [])],
                properties_by_adsorbate.get(item.id, []),
                synonyms_by_adsorbate.get(item.id, []),
            )
            for item in items
        ]

    # -------------------------------------------------------------------------
    def _chemical_view(self, session: Session, item: Adsorbate) -> dict[str, Any]:
        return self._chemical_views(session, [item])[0]

    # -------------------------------------------------------------------------
    @staticmethod
    def _chemical_payload(
        item: Adsorbate,
        identifiers: list[dict[str, Any]],
        properties: list[tuple[ChemicalProperty, str]],
        synonyms: list[str],
    ) -> dict[str, Any]:
        property_map = {prop.key: prop for prop, _ in properties}
        pubchem = next(
            (identifier for identifier in identifiers if identifier["source"] == "pubchem"),
            None,
        )
        cid = pubchem["external_id"] if pubchem else None
        preferred = property_map.get("preferred_name")
        connectivity = property_map.get("connectivity_smiles")
        has_3d = property_map.get("pubchem_3d_available") is not None
        return {
            "id": item.id,
            "name": item.name,
            "preferred_name": preferred.value_text if preferred else None,
            "formula": item.formula,
            "molecular_weight": item.molar_mass_g_mol,
            "inchi": item.inchi,
            "inchi_key": item.inchi_key,
            "connectivity_smiles": connectivity.value_text if connectivity else None,
            "smiles": item.smiles,
            "pubchem_cid": cid,
            "synonyms": synonyms,
            "properties": [
                {
                    "key": prop.key,
                    "value_number": prop.value_number,
                    "value_text": prop.value_text,
                    "unit": prop.unit,
                    "source": source_key,
                }
                for prop, source_key in properties
                if prop.key not in {"preferred_name", "connectivity_smiles", "pubchem_3d_available"}
            ],
            "external_identifiers": identifiers,
            "structure_2d_url": (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=300x300"
                if cid
                else None
            ),
            "conformer_3d_url": (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/SDF?record_type=3d"
                if cid and has_3d
                else None
            ),
            "retrieved_at": max(
                (
                    identifier["retrieved_at"]
                    for identifier in identifiers
                    if identifier["retrieved_at"]
                ),
                default=None,
            ),
        }

    # -------------------------------------------------------------------------
    def list_structures(
        self,
        *,
        page: int,
        page_size: int,
        query_text: str | None = None,
        source: str | None = None,
        linked_only: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self.database.session_factory() as session:
            query = select(Structure)
            if query_text:
                pattern = f"%{query_text.strip()}%"
                query = query.where(
                    or_(Structure.name.ilike(pattern), Structure.formula.ilike(pattern))
                )
            if linked_only is not None:
                query = query.where(
                    Structure.adsorbent_id.is_not(None)
                    if linked_only
                    else Structure.adsorbent_id.is_(None)
                )
            if source:
                ids = (
                    select(StructureSourceRecord.structure_id)
                    .join(
                        SourceRecord,
                        SourceRecord.id == StructureSourceRecord.source_record_id,
                    )
                    .join(DataSource, DataSource.id == SourceRecord.source_id)
                    .where(DataSource.key == source)
                )
                query = query.where(Structure.id.in_(ids))
            total = int(
                session.scalar(select(func.count()).select_from(query.subquery())) or 0
            )
            rows = session.scalars(
                query.order_by(Structure.updated_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            ).all()
            return self._structure_views(session, rows, include_atoms=False), total

    # -------------------------------------------------------------------------
    def get_structure(self, structure_id: int) -> dict[str, Any]:
        with self.database.session_factory() as session:
            item = session.get(Structure, structure_id)
            if item is None:
                raise LookupError(f"Structure {structure_id} was not found.")
            return self._structure_view(session, item, include_atoms=True)

    # -------------------------------------------------------------------------
    def _structure_views(
        self,
        session: Session,
        items: list[Structure],
        *,
        include_atoms: bool,
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        item_ids = [item.id for item in items]
        provenance = self._provenance_entries(session, "structure", item_ids)
        material_ids = [item.adsorbent_id for item in items if item.adsorbent_id is not None]
        material_names = {
            int(material_id): str(material_name)
            for material_id, material_name in session.execute(
                select(Adsorbent.id, Adsorbent.name).where(Adsorbent.id.in_(material_ids))
            )
        }
        atom_counts = {
            int(structure_id): int(count)
            for structure_id, count in session.execute(
                select(StructureAtom.structure_id, func.count(StructureAtom.id))
                .where(StructureAtom.structure_id.in_(item_ids))
                .group_by(StructureAtom.structure_id)
            )
        }
        atoms_by_structure: dict[int, list[dict[str, Any]]] = {}
        if include_atoms:
            atoms = session.scalars(
                select(StructureAtom)
                .where(StructureAtom.structure_id.in_(item_ids))
                .order_by(StructureAtom.structure_id, StructureAtom.sequence_index)
            ).all()
            for atom in atoms:
                atoms_by_structure.setdefault(atom.structure_id, []).append(
                    {
                        "sequence_index": atom.sequence_index,
                        "label": atom.label,
                        "element": atom.element,
                        "fractional_x": atom.fractional_x,
                        "fractional_y": atom.fractional_y,
                        "fractional_z": atom.fractional_z,
                        "occupancy": atom.occupancy,
                    }
                )
        primary_record_ids = [
            entries[0][0]
            for entries in provenance.values()
            if entries
        ]
        references = self._references_for_records(session, primary_record_ids)

        result: list[dict[str, Any]] = []
        for item in items:
            entries = provenance.get(item.id, [])
            source_record_id, primary = entries[0] if entries else (None, None)
            result.append(
                {
                    "id": item.id,
                    "source": primary["source"] if primary else "local",
                    "external_id": primary["external_id"] if primary else str(item.id),
                    "source_url": primary["source_url"] if primary else None,
                    "material_id": item.adsorbent_id,
                    "material_name": material_names.get(item.adsorbent_id)
                    if item.adsorbent_id is not None
                    else None,
                    "name": item.name,
                    "formula": item.formula,
                    "format": item.format,
                    "content_sha256": item.content_sha256,
                    "space_group": item.space_group,
                    "space_group_number": item.space_group_number,
                    "cell_a_angstrom": item.cell_a_angstrom,
                    "cell_b_angstrom": item.cell_b_angstrom,
                    "cell_c_angstrom": item.cell_c_angstrom,
                    "cell_alpha_deg": item.cell_alpha_deg,
                    "cell_beta_deg": item.cell_beta_deg,
                    "cell_gamma_deg": item.cell_gamma_deg,
                    "cell_volume_angstrom3": item.cell_volume_angstrom3,
                    "has_coordinates": item.has_coordinates,
                    "atom_count": atom_counts.get(item.id, 0),
                    "doi": references.get(source_record_id) if source_record_id else None,
                    "retrieved_at": primary["retrieved_at"] if primary else None,
                    "atoms": atoms_by_structure.get(item.id, []),
                }
            )
        return result

    # -------------------------------------------------------------------------
    def _structure_view(
        self, session: Session, item: Structure, *, include_atoms: bool
    ) -> dict[str, Any]:
        return self._structure_views(session, [item], include_atoms=include_atoms)[0]


__all__ = ["PublicDataRepository", "SOURCE_DEFINITIONS"]
