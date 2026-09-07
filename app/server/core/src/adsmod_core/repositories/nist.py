from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import func, select

from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.datasets import DatasetRepository
from adsmod_core.repositories.materials import MaterialRepository
from adsmod_core.repositories.public_data import PublicDataRepository
from adsmod_core.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    Isotherm,
    IsothermComponent,
    Observation,
)
from adsmod_core.repositories.schemas.types import normalize_identity

NIST_DATASET_NAME = "NIST ISODB"

###############################################################################
class NISTRepository:
    """Own canonical NIST persistence and query operations."""

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        database: DatabaseManager,
        datasets: DatasetRepository,
        materials: MaterialRepository,
        public_data: PublicDataRepository | None = None,
    ) -> None:
        self.database = database
        self.datasets = datasets
        self.materials = materials
        self.public_data = public_data

    # -------------------------------------------------------------------------
    def list_nist_experiment_ids(self) -> set[str]:
        with self.database.session_factory() as session:
            values = session.scalars(
                select(Isotherm.external_key)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            return {value.casefold() for value in values}

    # -------------------------------------------------------------------------
    def list_adsorbate_inchi_keys(self) -> set[str]:
        with self.database.session_factory() as session:
            values = session.scalars(
                select(Adsorbate.inchi_key)
                .join(
                    IsothermComponent,
                    IsothermComponent.adsorbate_id == Adsorbate.id,
                )
                .join(Isotherm, Isotherm.id == IsothermComponent.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(
                    Dataset.source == "nist",
                    Adsorbate.inchi_key.is_not(None),
                )
                .distinct()
            )
            return {str(value).casefold() for value in values}

    # -------------------------------------------------------------------------
    def list_adsorbent_hash_keys(self) -> set[str]:
        with self.database.session_factory() as session:
            values = session.scalars(
                select(Adsorbent.external_identifier)
                .join(Isotherm, Isotherm.adsorbent_id == Adsorbent.id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(
                    Dataset.source == "nist",
                    Adsorbent.external_identifier.is_not(None),
                )
                .distinct()
            )
            return {str(value).casefold() for value in values}

    # -------------------------------------------------------------------------
    def count_local_records_by_category(self) -> dict[str, int]:
        with self.database.session_factory() as session:
            experiments = session.scalar(
                select(func.count(Isotherm.id))
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            guests = session.scalar(
                select(func.count(func.distinct(Adsorbate.id)))
                .join(
                    IsothermComponent,
                    IsothermComponent.adsorbate_id == Adsorbate.id,
                )
                .join(Isotherm, Isotherm.id == IsothermComponent.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            hosts = session.scalar(
                select(func.count(func.distinct(Adsorbent.id)))
                .join(Isotherm, Isotherm.adsorbent_id == Adsorbent.id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
        return {
            "experiments": int(experiments or 0),
            "guest": int(guests or 0),
            "host": int(hosts or 0),
        }

    # -------------------------------------------------------------------------
    def count_nist_rows(self) -> dict[str, int]:
        with self.database.session_factory() as session:
            experiments_count = session.scalar(
                select(func.count(Isotherm.id))
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            single_component_rows = session.scalar(
                select(func.count(Observation.id))
                .join(Isotherm, Isotherm.id == Observation.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            mixture_ids = (
                select(IsothermComponent.isotherm_id)
                .join(Isotherm, Isotherm.id == IsothermComponent.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
                .group_by(IsothermComponent.isotherm_id)
                .having(func.count(IsothermComponent.id) > 1)
                .subquery()
            )
            binary_mixture_rows = session.scalar(
                select(func.count()).select_from(mixture_ids)
            )
            guest_rows = session.scalar(
                select(func.count(func.distinct(Adsorbate.id)))
                .join(IsothermComponent, IsothermComponent.adsorbate_id == Adsorbate.id)
                .join(Observation, Observation.component_id == IsothermComponent.id)
                .join(Isotherm, Isotherm.id == Observation.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
            host_rows = session.scalar(
                select(func.count(func.distinct(Adsorbent.id)))
                .join(Isotherm, Isotherm.adsorbent_id == Adsorbent.id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
            )
        return {
            "experiments_count": int(experiments_count or 0),
            "single_component_rows": int(single_component_rows or 0),
            "binary_mixture_rows": int(binary_mixture_rows or 0),
            "guest_rows": int(guest_rows or 0),
            "host_rows": int(host_rows or 0),
        }

    # -------------------------------------------------------------------------
    def load_adsorption_datasets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        with self.database.session_factory() as session:
            rows = session.execute(
                select(
                    Isotherm.external_key,
                    Isotherm.temperature_k,
                    Adsorbent.name.label("adsorbent"),
                    Adsorbate.name.label("adsorbate"),
                    Observation.pressure_canonical.label("pressure"),
                    Observation.uptake_mol_kg.label("adsorbed_amount"),
                )
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
                .join(Observation, Observation.isotherm_id == Isotherm.id)
                .join(
                    IsothermComponent,
                    IsothermComponent.id == Observation.component_id,
                )
                .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
                .where(Dataset.source == "nist")
                .order_by(Isotherm.id, Observation.sequence_index)
            ).mappings()
            guests = session.scalars(
                select(Adsorbate)
                .join(
                    IsothermComponent,
                    IsothermComponent.adsorbate_id == Adsorbate.id,
                )
                .join(Isotherm, Isotherm.id == IsothermComponent.isotherm_id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
                .distinct()
                .order_by(Adsorbate.id)
            ).all()
            hosts = session.scalars(
                select(Adsorbent)
                .join(Isotherm, Isotherm.adsorbent_id == Adsorbent.id)
                .join(Dataset, Dataset.id == Isotherm.dataset_id)
                .where(Dataset.source == "nist")
                .distinct()
                .order_by(Adsorbent.id)
            ).all()
        adsorption = pd.DataFrame([dict(row) for row in rows])
        guest = pd.DataFrame(
            [
                {
                    "name": item.name,
                    "InChIKey": item.inchi_key,
                    "molecular_weight": item.molar_mass_g_mol,
                    "molecular_formula": item.formula,
                    "smile_code": item.smiles,
                }
                for item in guests
            ]
        )
        host = pd.DataFrame(
            [
                {
                    "name": item.name,
                    "hashkey": item.external_identifier,
                    "molecular_weight": item.molar_mass_g_mol,
                    "molecular_formula": item.formula,
                    "smile_code": item.smiles,
                }
                for item in hosts
            ]
        )
        return adsorption, guest, host

    # -------------------------------------------------------------------------
    def save_materials(
        self,
        guest_records: list[dict[str, Any]],
        host_records: list[dict[str, Any]],
    ) -> None:
        if guest_records:
            self.materials.upsert_adsorbates(guest_records)
        if host_records:
            self.materials.upsert_adsorbents(host_records)
        if self.public_data is None:
            return

        guest_ids = self.materials.adsorbate_ids(
            str(record["key"]) for record in guest_records
        )
        for record in guest_records:
            inchi_key = str(record.get("inchi_key") or "").strip()
            if not inchi_key:
                continue
            identifier = guest_ids.get(str(record["key"]))
            if identifier is None:
                continue
            self.public_data.link_adsorbate_record(
                source_key="nist",
                adsorbate_id=identifier,
                external_id=inchi_key,
                source_url=f"https://adsorption.nist.gov/isodb/api/gas/{inchi_key}.json",
            )

        host_ids = self.materials.adsorbent_ids(
            str(record["key"]) for record in host_records
        )
        for record in host_records:
            external_identifier = str(record.get("external_identifier") or "").strip()
            if not external_identifier:
                continue
            identifier = host_ids.get(str(record["key"]))
            if identifier is None:
                continue
            self.public_data.link_adsorbent_record(
                source_key="nist",
                adsorbent_id=identifier,
                external_id=external_identifier,
                source_url=(
                    "https://adsorption.nist.gov/matdb/api/material/"
                    f"{external_identifier}.json"
                ),
            )

    # -------------------------------------------------------------------------
    def save_experiments(
        self, experiments: list[dict[str, Any]], _replace: bool = False
    ) -> None:
        existing = self.list_nist_experiment_ids()
        with self.database.session_factory() as session:
            collection_id = session.scalar(
                select(Dataset.id).where(
                    Dataset.source == "nist",
                    Dataset.normalized_name == normalize_identity(NIST_DATASET_NAME),
                )
            )
        if collection_id is None:
            collection_id = self.datasets.persist_canonical(
                name=NIST_DATASET_NAME,
                source="nist",
                provenance={
                    "repository": NIST_DATASET_NAME,
                    "ingestion": "deterministic known-schema",
                },
                experiments=[],
            )
        for experiment in experiments:
            source_id = str(experiment["external_key"])
            if source_id.casefold() in existing:
                continue
            self.datasets.persist_canonical(
                name=NIST_DATASET_NAME,
                source="nist",
                provenance={
                    "repository": NIST_DATASET_NAME,
                    "source_identifier": source_id,
                    "ingestion": "deterministic known-schema",
                },
                experiments=[experiment],
                dataset_id=collection_id,
            )
            if self.public_data is not None:
                self.public_data.link_isotherm_record(
                    source_key="nist",
                    dataset_id=collection_id,
                    external_id=source_id,
                    source_url="https://adsorption.nist.gov/",
                    raw_metadata={"collection": NIST_DATASET_NAME},
                )
            existing.add(source_id.casefold())
