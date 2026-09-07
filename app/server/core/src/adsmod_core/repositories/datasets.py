from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload
import numpy as np

from adsmod_core.repositories.database.manager import DatabaseManager
from adsmod_core.repositories.schemas.models import (
    Adsorbate,
    Adsorbent,
    Dataset,
    DatasetImport,
    Isotherm,
    IsothermComponent,
    Observation,
)
from adsmod_core.repositories.schemas.types import normalize_identity

###############################################################################
def stable_material_key(kind: str, name: str, external: str | None = None) -> str:
    identity = normalize_identity(external or name)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"{kind}:{digest}"

###############################################################################
class DatasetRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    def list_summaries(self) -> list[dict[str, Any]]:
        statement = (
            select(
                Dataset,
                func.count(func.distinct(Isotherm.id)).label("experiment_count"),
                func.count(Observation.id).label("observation_count"),
            )
            .outerjoin(Isotherm, Isotherm.dataset_id == Dataset.id)
            .outerjoin(Observation, Observation.isotherm_id == Isotherm.id)
            .group_by(Dataset.id)
            .order_by(Dataset.created_at.desc(), Dataset.id.desc())
        )
        with self.database.session_factory() as session:
            rows = session.execute(statement).all()
            return [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "source": dataset.source,
                    "created_at": dataset.created_at.isoformat(),
                    "experiment_count": int(experiment_count or 0),
                    "observation_count": int(observation_count or 0),
                    "tags": list(dataset.tags or []),
                    "description": dataset.description or "",
                }
                for dataset, experiment_count, observation_count in rows
            ]

    # -------------------------------------------------------------------------
    def summary(self, dataset_id: int) -> dict[str, Any]:
        for item in self.list_summaries():
            if item["id"] == dataset_id:
                return item
        raise LookupError(f"Dataset {dataset_id} does not exist.")

    # -------------------------------------------------------------------------
    def observation_frame(self, dataset_id: int) -> pd.DataFrame:
        """Return one canonical row per persisted adsorption observation."""
        statement = (
            select(
                Dataset.id.label("dataset_id"),
                Dataset.name.label("dataset_name"),
                Isotherm.external_key.label("experiment"),
                Isotherm.temperature_k.label("temperature"),
                Adsorbent.name.label("adsorbent_name"),
                Adsorbate.name.label("adsorbate_name"),
                Adsorbate.molar_mass_g_mol.label("adsorbate_molecular_weight"),
                Adsorbate.smiles.label("adsorbate_SMILE"),
                Observation.pressure_canonical.label("pressure"),
                Observation.uptake_mol_kg.label("adsorbed_amount"),
            )
            .join(Isotherm, Isotherm.dataset_id == Dataset.id)
            .join(Adsorbent, Adsorbent.id == Isotherm.adsorbent_id)
            .join(Observation, Observation.isotherm_id == Isotherm.id)
            .join(IsothermComponent, IsothermComponent.id == Observation.component_id)
            .join(Adsorbate, Adsorbate.id == IsothermComponent.adsorbate_id)
            .where(Dataset.id == dataset_id)
            .order_by(Isotherm.id, Observation.sequence_index, Observation.id)
        )
        with self.database.session_factory() as session:
            if session.get(Dataset, dataset_id) is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            rows = session.execute(statement).mappings()
            return pd.DataFrame([dict(row) for row in rows])

    # -------------------------------------------------------------------------
    def rename(self, dataset_id: int, name: str) -> None:
        normalized = normalize_identity(name)
        with self.database.transaction() as session:
            existing = session.scalar(
                select(Dataset.id).where(
                    Dataset.normalized_name == normalized, Dataset.id != dataset_id
                )
            )
            if existing is not None:
                raise ValueError(f"A dataset named '{name}' already exists.")
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            dataset.name = name.strip()
            dataset.normalized_name = normalized

    # -------------------------------------------------------------------------
    def update_metadata(
        self, dataset_id: int, tags: Iterable[str], description: str
    ) -> None:
        with self.database.transaction() as session:
            dataset = session.get(Dataset, dataset_id)
            if dataset is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            dataset.tags = sorted({tag.strip() for tag in tags if tag.strip()})
            dataset.description = description.strip()

    # -------------------------------------------------------------------------
    def delete(self, dataset_id: int) -> None:
        with self.database.transaction() as session:
            result = session.execute(delete(Dataset).where(Dataset.id == dataset_id))
            if not result.rowcount:
                raise LookupError(f"Dataset {dataset_id} does not exist.")

    # -------------------------------------------------------------------------
    def experiments(self, dataset_id: int) -> list[dict[str, Any]]:
        statement = (
            select(Isotherm)
            .where(Isotherm.dataset_id == dataset_id)
            .options(
                selectinload(Isotherm.adsorbent),
                selectinload(Isotherm.components).selectinload(
                    IsothermComponent.adsorbate
                ),
                selectinload(Isotherm.observations),
            )
            .order_by(Isotherm.name, Isotherm.id)
        )
        with self.database.session_factory() as session:
            isotherms = list(session.scalars(statement).unique())
            if not isotherms and session.get(Dataset, dataset_id) is None:
                raise LookupError(f"Dataset {dataset_id} does not exist.")
            output: list[dict[str, Any]] = []
            for isotherm in isotherms:
                components = sorted(isotherm.components, key=lambda item: item.position)
                component_names = [item.adsorbate.name for item in components]
                eligible = len(components) == 1 and len(isotherm.observations) >= 2
                reason = None
                if len(components) != 1:
                    reason = (
                        "The available theoretical models are single-component models."
                    )
                elif len(isotherm.observations) < 2:
                    reason = "At least two observations are required."
                output.append(
                    {
                        "id": isotherm.id,
                        "dataset_id": dataset_id,
                        "external_key": isotherm.external_key,
                        "name": isotherm.name,
                        "adsorbent": isotherm.adsorbent.name,
                        "adsorbates": component_names,
                        "temperature_k": isotherm.temperature_k,
                        "pressure_basis": isotherm.pressure_basis,
                        "observation_count": len(isotherm.observations),
                        "fitting_eligible": eligible,
                        "ineligibility_reason": reason,
                    }
                )
            return output

    # -------------------------------------------------------------------------
    def observations(
        self, dataset_id: int, isotherm_id: int, offset: int = 0, limit: int = 100
    ) -> tuple[list[dict[str, Any]], int]:
        base = (
            select(Observation)
            .join(Isotherm, Isotherm.id == Observation.isotherm_id)
            .where(
                Isotherm.dataset_id == dataset_id,
                Observation.isotherm_id == isotherm_id,
            )
        )
        with self.database.session_factory() as session:
            total = int(
                session.scalar(select(func.count()).select_from(base.subquery())) or 0
            )
            rows = list(
                session.scalars(
                    base.order_by(
                        Observation.pressure_canonical,
                        Observation.sequence_index,
                        Observation.id,
                    )
                    .offset(offset)
                    .limit(limit)
                )
            )
            if not rows and session.get(Isotherm, isotherm_id) is None:
                raise LookupError(f"Isotherm {isotherm_id} does not exist.")
            return (
                [
                    {
                        "id": row.id,
                        "sequence_index": row.sequence_index,
                        "source_row": row.source_row,
                        "pressure_original": row.pressure_original,
                        "pressure_original_unit": row.pressure_original_unit,
                        "pressure_canonical": row.pressure_canonical,
                        "pressure_canonical_unit": row.pressure_canonical_unit,
                        "uptake_original": row.uptake_original,
                        "uptake_original_unit": row.uptake_original_unit,
                        "uptake_mol_kg": row.uptake_mol_kg,
                        "uptake_stddev_mol_kg": row.uptake_stddev_mol_kg,
                    }
                    for row in rows
                ],
                total,
            )

    # -------------------------------------------------------------------------
    def fitting_series(self, dataset_id: int, isotherm_id: int) -> dict[str, Any]:
        statement = (
            select(Isotherm)
            .where(Isotherm.id == isotherm_id, Isotherm.dataset_id == dataset_id)
            .options(
                selectinload(Isotherm.dataset),
                selectinload(Isotherm.adsorbent),
                selectinload(Isotherm.components).selectinload(
                    IsothermComponent.adsorbate
                ),
                selectinload(Isotherm.observations),
            )
        )
        with self.database.session_factory() as session:
            isotherm = session.scalar(statement)
            if isotherm is None:
                raise LookupError(
                    f"Isotherm {isotherm_id} does not belong to dataset {dataset_id}."
                )
            components = sorted(isotherm.components, key=lambda item: item.position)
            if len(components) != 1:
                raise ValueError(
                    "The available theoretical models require a single-component isotherm."
                )
            component = components[0]
            observations = sorted(
                (
                    item
                    for item in isotherm.observations
                    if item.component_id == component.id
                ),
                key=lambda item: (
                    item.pressure_canonical,
                    item.sequence_index,
                    item.id,
                ),
            )
            if len(observations) < 2:
                raise ValueError(
                    "At least two canonical observations are required for fitting."
                )
            if isotherm.duplicate_policy == "average":
                grouped: dict[float, list[Observation]] = {}
                for item in observations:
                    grouped.setdefault(float(item.pressure_canonical), []).append(item)
                averaged: list[dict[str, Any]] = []
                source_ids: list[int] = []
                for index, (pressure, group) in enumerate(
                    sorted(grouped.items()), start=0
                ):
                    source_ids.extend(item.id for item in group)
                    values = [float(item.uptake_mol_kg) for item in group]
                    sigmas = [item.uptake_stddev_mol_kg for item in group]
                    sigma = (
                        float(np.sqrt(np.sum(np.square(sigmas))) / len(sigmas))
                        if sigmas
                        and all(value is not None and value > 0 for value in sigmas)
                        else None
                    )
                    averaged.append((pressure, float(np.mean(values)), sigma))
                pressure_values = [item[0] for item in averaged]
                uptake_values = [item[1] for item in averaged]
                sigma_values = [item[2] for item in averaged]
            else:
                source_ids = [item.id for item in observations]
                pressure_values = [item.pressure_canonical for item in observations]
                uptake_values = [item.uptake_mol_kg for item in observations]
                sigma_values = [item.uptake_stddev_mol_kg for item in observations]
            return {
                "dataset_id": isotherm.dataset.id,
                "dataset_name": isotherm.dataset.name,
                "isotherm_id": isotherm.id,
                "isotherm_name": isotherm.name,
                "component_id": component.id,
                "adsorbent": isotherm.adsorbent.name,
                "adsorbate": component.adsorbate.name,
                "adsorbate_molar_mass_g_mol": component.adsorbate.molar_mass_g_mol,
                "temperature_k": isotherm.temperature_k,
                "pressure_basis": isotherm.pressure_basis,
                "saturation_pressure_pa": isotherm.saturation_pressure_pa,
                "observation_ids": source_ids,
                "pressure": pressure_values,
                "uptake": uptake_values,
                "uptake_stddev": sigma_values,
                "duplicate_policy": isotherm.duplicate_policy,
            }

    # -------------------------------------------------------------------------
    def persist_canonical(
        self,
        *,
        name: str,
        source: str,
        provenance: dict[str, Any],
        experiments: list[dict[str, Any]],
        import_manifest: dict[str, Any] | None = None,
        dataset_id: int | None = None,
    ) -> int:
        normalized_name = normalize_identity(name)
        with self.database.transaction() as session:
            if dataset_id is None:
                if (
                    session.scalar(
                        select(Dataset.id).where(
                            Dataset.normalized_name == normalized_name
                        )
                    )
                    is not None
                ):
                    raise ValueError(f"A dataset named '{name}' already exists.")
                dataset = Dataset(
                    name=name.strip(),
                    normalized_name=normalized_name,
                    source=source,
                    provenance=provenance,
                )
                session.add(dataset)
                session.flush()
            else:
                dataset = session.get(Dataset, dataset_id)
                if dataset is None or dataset.source != source:
                    raise LookupError(
                        f"Canonical {source} dataset {dataset_id} does not exist."
                    )

            if import_manifest is not None:
                session.add(DatasetImport(dataset_id=dataset.id, **import_manifest))

            adsorbates_by_key: dict[str, Adsorbate] = {}
            adsorbents_by_key: dict[str, Adsorbent] = {}
            for record in experiments:
                adsorbent_record = record["adsorbent"]
                adsorbent_key = adsorbent_record.get("key") or stable_material_key(
                    "adsorbent",
                    adsorbent_record["name"],
                    adsorbent_record.get("external_identifier"),
                )
                adsorbent = adsorbents_by_key.get(adsorbent_key)
                if adsorbent is None:
                    adsorbent = session.scalar(
                        select(Adsorbent).where(Adsorbent.key == adsorbent_key)
                    )
                if adsorbent is None:
                    adsorbent = Adsorbent(
                        key=adsorbent_key,
                        name=adsorbent_record["name"].strip(),
                        external_identifier=adsorbent_record.get("external_identifier"),
                    )
                    session.add(adsorbent)
                    session.flush()
                adsorbents_by_key[adsorbent_key] = adsorbent

                isotherm = Isotherm(
                    dataset_id=dataset.id,
                    external_key=record["external_key"],
                    name=record["name"],
                    adsorbent_id=adsorbent.id,
                    temperature_original=record["temperature_original"],
                    temperature_original_unit=record["temperature_original_unit"],
                    temperature_k=record["temperature_k"],
                    pressure_basis=record["pressure_basis"],
                    duplicate_policy=record.get("duplicate_policy", "reject"),
                    saturation_pressure_pa=record.get("saturation_pressure_pa"),
                    conditions=record.get("conditions", {}),
                    provenance=record.get("provenance", {}),
                )
                session.add(isotherm)
                session.flush()

                component_by_name: dict[str, IsothermComponent] = {}
                for position, adsorbate_record in enumerate(
                    record["adsorbates"], start=1
                ):
                    adsorbate_key = adsorbate_record.get("key") or stable_material_key(
                        "adsorbate",
                        adsorbate_record["name"],
                        adsorbate_record.get("inchi_key"),
                    )
                    adsorbate = adsorbates_by_key.get(adsorbate_key)
                    if adsorbate is None:
                        adsorbate = session.scalar(
                            select(Adsorbate).where(Adsorbate.key == adsorbate_key)
                        )
                    if adsorbate is None:
                        adsorbate = Adsorbate(
                            key=adsorbate_key,
                            name=adsorbate_record["name"].strip(),
                            inchi_key=adsorbate_record.get("inchi_key"),
                            inchi=adsorbate_record.get("inchi"),
                            formula=adsorbate_record.get("formula"),
                            molar_mass_g_mol=adsorbate_record.get("molar_mass_g_mol"),
                            smiles=adsorbate_record.get("smiles"),
                        )
                        session.add(adsorbate)
                        session.flush()
                    elif adsorbate_record.get("smiles") and not adsorbate.smiles:
                        adsorbate.smiles = adsorbate_record["smiles"]
                    adsorbates_by_key[adsorbate_key] = adsorbate
                    component = IsothermComponent(
                        isotherm_id=isotherm.id,
                        position=position,
                        adsorbate_id=adsorbate.id,
                        mole_fraction=adsorbate_record.get("mole_fraction"),
                    )
                    session.add(component)
                    session.flush()
                    component_by_name[normalize_identity(adsorbate_record["name"])] = (
                        component
                    )

                for observation in record["observations"]:
                    component = component_by_name[
                        normalize_identity(observation["adsorbate"])
                    ]
                    session.add(
                        Observation(
                            isotherm_id=isotherm.id,
                            component_id=component.id,
                            sequence_index=observation["sequence_index"],
                            source_row=observation.get("source_row"),
                            pressure_original=observation["pressure_original"],
                            pressure_original_unit=observation[
                                "pressure_original_unit"
                            ],
                            pressure_canonical=observation["pressure_canonical"],
                            pressure_canonical_unit=observation[
                                "pressure_canonical_unit"
                            ],
                            uptake_original=observation["uptake_original"],
                            uptake_original_unit=observation["uptake_original_unit"],
                            uptake_mol_kg=observation["uptake_mol_kg"],
                            uptake_stddev_mol_kg=observation.get(
                                "uptake_stddev_mol_kg"
                            ),
                            conversion_metadata=observation.get(
                                "conversion_metadata", {}
                            ),
                            extra_metadata=observation.get("extra_metadata", {}),
                        )
                    )
            session.flush()
            return dataset.id
