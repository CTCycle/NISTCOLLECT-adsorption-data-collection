from __future__ import annotations

from typing import Any

import pandas as pd

from adsmod_common.units import UnitRegistry, parse_number
from adsmod_core.common.utils.logger import logger
from adsmod_core.repositories.datasets import stable_material_key

###############################################################################
def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()

###############################################################################
class NISTCanonicalMapper:
    """Map known NIST provider frames to canonical persistence records."""

    # -------------------------------------------------------------------------
    def material_records(
        self, frame: pd.DataFrame | None, kind: str
    ) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        records: list[dict[str, Any]] = []
        for row in frame.where(frame.notna(), None).to_dict(orient="records"):
            name = _text(row.get("name"))
            if not name:
                continue
            external = _text(
                row.get("InChIKey") if kind == "adsorbate" else row.get("hashkey")
            )
            records.append(
                {
                    "key": stable_material_key(kind, name, external or None),
                    "name": name,
                    **(
                        {"inchi_key": external or None}
                        if kind == "adsorbate"
                        else {"external_identifier": external or None}
                    ),
                    "formula": _text(row.get("molecular_formula")) or None,
                    "molar_mass_g_mol": (
                        float(row["molecular_weight"])
                        if row.get("molecular_weight") is not None
                        else None
                    ),
                    "smiles": _text(row.get("smile_code")) or None,
                }
            )
        return records

    # -------------------------------------------------------------------------
    def experiment_records(
        self,
        single_component: pd.DataFrame | None,
        binary_mixture: pd.DataFrame | None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source_id, rows, structure in self._grouped_experiments(
            single_component, binary_mixture
        ):
            try:
                record = (
                    self._single_experiment(source_id, rows)
                    if structure == "single"
                    else self._binary_experiment(source_id, rows)
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping NIST experiment %s: cannot map measurements to canonical units: %s",
                    source_id,
                    exc,
                )
                continue
            records.append(record)
        return records

    # -------------------------------------------------------------------------
    @staticmethod
    def _grouped_experiments(
        single_component: pd.DataFrame | None,
        binary_mixture: pd.DataFrame | None,
    ) -> list[tuple[str, pd.DataFrame, str]]:
        grouped: list[tuple[str, pd.DataFrame, str]] = []
        if single_component is not None and not single_component.empty:
            grouped.extend(
                (str(name), rows, "single")
                for name, rows in single_component.groupby("name", sort=False)
            )
        if binary_mixture is not None and not binary_mixture.empty:
            grouped.extend(
                (str(name), rows, "binary")
                for name, rows in binary_mixture.groupby("name", sort=False)
            )
        return grouped

    # -------------------------------------------------------------------------
    @staticmethod
    def _single_experiment(source_id: str, rows: pd.DataFrame) -> dict[str, Any]:
        first = rows.iloc[0]
        pressure_unit = _text(first["pressure_units"])
        uptake_unit = _text(first["adsorption_units"])
        temperature = parse_number(first["temperature"], ".")
        adsorbent = _text(first["adsorbent"])
        adsorbate = _text(first["adsorbate"])
        molar_mass = (
            float(first.get("adsorbate_molecular_weight"))
            if first.get("adsorbate_molecular_weight") not in (None, "")
            else None
        )
        observations: list[dict[str, Any]] = []
        for index, row in enumerate(rows.itertuples(index=False), start=0):
            pressure_value = parse_number(getattr(row, "pressure"), ".")
            uptake_value = parse_number(getattr(row, "adsorbed_amount"), ".")
            pressure = UnitRegistry.convert_pressure(
                pressure_value, pressure_unit, "absolute"
            )
            uptake = UnitRegistry.convert_uptake(uptake_value, uptake_unit, molar_mass)
            observations.append(
                {
                    "adsorbate": adsorbate,
                    "sequence_index": index,
                    "source_row": index + 1,
                    "pressure_original": pressure_value,
                    "pressure_original_unit": pressure.original_unit,
                    "pressure_canonical": pressure.canonical_value,
                    "pressure_canonical_unit": pressure.canonical_unit,
                    "uptake_original": uptake_value,
                    "uptake_original_unit": uptake.original_unit,
                    "uptake_mol_kg": uptake.canonical_value,
                    "conversion_metadata": {
                        "pressure": pressure.rule,
                        "uptake": uptake.rule,
                        "source": "NIST ISODB known schema",
                    },
                }
            )
        return {
            "external_key": source_id,
            "name": source_id,
            "adsorbent": {"name": adsorbent},
            "adsorbates": [{"name": adsorbate, "molar_mass_g_mol": molar_mass}],
            "temperature_original": temperature,
            "temperature_original_unit": "K",
            "temperature_k": temperature,
            "pressure_basis": "absolute",
            "conditions": {},
            "provenance": {
                "repository": "NIST ISODB",
                "source_identifier": source_id,
            },
            "observations": observations,
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _binary_experiment(source_id: str, rows: pd.DataFrame) -> dict[str, Any]:
        first = rows.iloc[0]
        pressure_unit = _text(first["pressure_units"])
        uptake_unit = _text(first["adsorption_units"])
        temperature = parse_number(first["temperature"], ".")
        adsorbent = _text(first["adsorbent_name"])
        species = [_text(first["compound_1"]), _text(first["compound_2"])]
        observations: list[dict[str, Any]] = []
        for point_index, row in rows.reset_index(drop=True).iterrows():
            for position, species_name in enumerate(species, start=1):
                pressure_value = parse_number(row[f"compound_{position}_pressure"], ".")
                uptake_value = parse_number(row[f"compound_{position}_adsorption"], ".")
                pressure = UnitRegistry.convert_pressure(
                    pressure_value, pressure_unit, "partial"
                )
                uptake = UnitRegistry.convert_uptake(uptake_value, uptake_unit)
                observations.append(
                    {
                        "adsorbate": species_name,
                        "sequence_index": point_index,
                        "source_row": point_index + 1,
                        "pressure_original": pressure_value,
                        "pressure_original_unit": pressure.original_unit,
                        "pressure_canonical": pressure.canonical_value,
                        "pressure_canonical_unit": pressure.canonical_unit,
                        "uptake_original": uptake_value,
                        "uptake_original_unit": uptake.original_unit,
                        "uptake_mol_kg": uptake.canonical_value,
                        "conversion_metadata": {
                            "pressure": pressure.rule,
                            "uptake": uptake.rule,
                            "source": "NIST ISODB known schema",
                        },
                    }
                )
        return {
            "external_key": source_id,
            "name": source_id,
            "adsorbent": {"name": adsorbent},
            "adsorbates": [{"name": name} for name in species],
            "temperature_original": temperature,
            "temperature_original_unit": "K",
            "temperature_k": temperature,
            "pressure_basis": "partial",
            "conditions": {"mixture": True},
            "provenance": {
                "repository": "NIST ISODB",
                "source_identifier": source_id,
            },
            "observations": observations,
        }
