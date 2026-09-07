from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from adsmod_core.contracts.datasets import (
    ColumnDetection,
    ImportIssue,
    ImportMapping,
    ImportPreviewResponse,
    ImportValidationResponse,
    NormalizedExperimentPreview,
    NormalizedObservationPreview,
)
from adsmod_core.services.data.import_parser import (
    detect_structure,
    infer_column,
    infer_pressure_basis,
    parse_series_cell,
    read_tabular,
    safe_cell,
    source_hash,
)
from adsmod_common.units import (
    UnitConversionError,
    UnitRegistry,
    parse_number,
)

PARSER_VERSION = "2.0"
PREVIEW_ROWS = 12
EXPERIMENT_PREVIEW_POINTS = 8

###############################################################################
@dataclass
class ValidationBundle:
    response: ImportValidationResponse
    experiments: list[dict[str, Any]]

###############################################################################
class AdsorptionImportEngine:

    # -------------------------------------------------------------------------
    def preview(self, payload: bytes, filename: str | None) -> ImportPreviewResponse:
        frame = read_tabular(payload, filename)
        columns = [infer_column(name, frame[name]) for name in frame.columns]
        structure, confidence = detect_structure(columns)
        grouping = [
            column.name
            for column in columns
            if column.proposed_role == "experiment_id" and column.confidence >= 0.65
        ][:1]
        issues: list[ImportIssue] = []
        if structure == "ambiguous":
            issues.append(
                ImportIssue(
                    code="ambiguous_structure",
                    severity="confirmation",
                    message="The file structure could not be classified safely.",
                    remediation="Choose atomic or aggregated layout and map the measurement columns.",
                )
            )
        for role in ("pressure", "uptake", "temperature", "adsorbate", "adsorbent"):
            matches = [
                column
                for column in columns
                if column.proposed_role == role and column.confidence >= 0.55
            ]
            if not matches:
                issues.append(
                    ImportIssue(
                        code=f"unmapped_{role}",
                        severity="confirmation",
                        message=f"No reliable {role.replace('_', ' ')} column was detected.",
                        remediation="Map a source column or provide a dataset-level constant.",
                    )
                )
        return ImportPreviewResponse(
            filename=filename or "dataset",
            source_sha256=source_hash(payload),
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            detected_structure=structure,  # type: ignore[arg-type]
            structure_confidence=confidence,
            columns=columns,
            preview_rows=[
                {key: safe_cell(value) for key, value in row.items()}
                for row in frame.head(PREVIEW_ROWS).to_dict(orient="records")
            ],
            proposed_grouping_columns=grouping,
            proposed_pressure_basis=infer_pressure_basis(columns),  # type: ignore[arg-type]
            issues=issues,
            guidance=[
                "Atomic one-observation-per-row data is the recommended format.",
                "Aggregated formats are accepted only when pressure and uptake series can be paired unambiguously.",
                "Convert the source to atomic format if validation cannot establish a safe interpretation.",
            ],
        )

    # -------------------------------------------------------------------------
    def validate(
        self, payload: bytes, filename: str | None, mapping: ImportMapping
    ) -> ValidationBundle:
        frame = read_tabular(
            payload,
            filename,
            header_row=mapping.header_row,
            field_delimiter=mapping.field_delimiter,
            encoding=mapping.encoding,
            worksheet=mapping.worksheet,
        )
        issues: list[ImportIssue] = []
        unknown_columns = sorted(set(mapping.column_roles) - set(frame.columns))
        if unknown_columns:
            issues.extend(
                ImportIssue(
                    code="unknown_mapping_column",
                    severity="error",
                    column=column,
                    message=f"Mapped column '{column}' does not exist in the uploaded file.",
                )
                for column in unknown_columns
            )

        roles = {
            role: column
            for column, role in mapping.column_roles.items()
            if role not in {"ignore", "metadata"}
        }
        for required in ("pressure", "uptake", "temperature", "adsorbate", "adsorbent"):
            if required not in roles and required not in mapping.constants:
                issues.append(
                    ImportIssue(
                        code=f"missing_{required}",
                        severity="error",
                        message=f"A {required.replace('_', ' ')} column or constant is required.",
                    )
                )
        if not mapping.grouping_columns and not mapping.whole_file_grouping:
            issues.append(
                ImportIssue(
                    code="missing_grouping_column",
                    severity="error",
                    message="Select one or more grouping columns, or explicitly choose the entire file as one experiment.",
                )
            )
        for column in mapping.grouping_columns:
            if column not in frame.columns:
                issues.append(
                    ImportIssue(
                        code="missing_grouping_column",
                        severity="error",
                        column=column,
                        message=f"Grouping column '{column}' does not exist.",
                    )
                )
        if issues:
            return self._bundle(mapping, payload, [], issues)

        detected = {
            item.name: item
            for item in [infer_column(name, frame[name]) for name in frame]
        }
        grouped_records: dict[str, list[dict[str, Any]]] = {}
        metadata_by_group: dict[str, dict[str, Any]] = {}

        for row_offset, row in frame.iterrows():
            source_row = int(row_offset) + 2
            group_values = [
                str(row.get(column, "")).strip() for column in mapping.grouping_columns
            ]
            if mapping.whole_file_grouping:
                group_values = [
                    mapping.constants.get("experiment_name", mapping.dataset_name)
                ]
            if any(
                not value or value.casefold() in {"nan", "none"}
                for value in group_values
            ):
                issues.append(
                    ImportIssue(
                        code="missing_experiment_identifier",
                        severity="error",
                        source_row=source_row,
                        message="Experiment grouping values must not be empty.",
                    )
                )
                continue
            group_key = " | ".join(group_values)
            try:
                row_records, row_metadata = self._normalize_row(
                    row,
                    source_row,
                    roles,
                    mapping,
                    detected,
                )
            except (ValueError, UnitConversionError) as exc:
                issues.append(
                    ImportIssue(
                        code="invalid_row",
                        severity="error",
                        source_row=source_row,
                        experiment=group_key,
                        message=str(exc),
                    )
                )
                continue
            existing = metadata_by_group.get(group_key)
            if existing is not None:
                for key in (
                    "adsorbent",
                    "adsorbate",
                    "adsorbate_smiles",
                    "temperature_k",
                    "pressure_basis",
                ):
                    if existing[key] != row_metadata[key]:
                        issues.append(
                            ImportIssue(
                                code="inconsistent_experiment_metadata",
                                severity="error",
                                source_row=source_row,
                                experiment=group_key,
                                message=(
                                    f"Experiment '{group_key}' contains inconsistent {key.replace('_', ' ')} values."
                                ),
                                remediation="Correct the source data or include the differing field in the grouping selection.",
                            )
                        )
            else:
                metadata_by_group[group_key] = row_metadata
            grouped_records.setdefault(group_key, []).extend(row_records)

        experiments: list[dict[str, Any]] = []
        for group_key, observations in grouped_records.items():
            if group_key not in metadata_by_group:
                continue
            metadata = metadata_by_group[group_key]
            observations.sort(
                key=lambda item: (
                    item["pressure_canonical"],
                    item["source_row"] or 0,
                    item["sequence_index"],
                )
            )
            duplicate_pressures: dict[float, list[dict[str, Any]]] = {}
            for observation in observations:
                duplicate_pressures.setdefault(
                    observation["pressure_canonical"], []
                ).append(observation)
            repeated = {
                pressure: items
                for pressure, items in duplicate_pressures.items()
                if len(items) > 1
            }
            if repeated and mapping.duplicate_policy == "reject":
                issues.append(
                    ImportIssue(
                        code="duplicate_pressure",
                        severity="confirmation",
                        experiment=group_key,
                        message=(
                            f"Experiment '{group_key}' has repeated pressure values. "
                            "Keeping them weights those pressures as replicate observations."
                        ),
                        remediation="Choose the keep-replicates policy or correct duplicate rows.",
                    )
                )
            elif repeated and mapping.duplicate_policy == "average":
                issues.append(
                    ImportIssue(
                        code="duplicates_average_on_fit",
                        severity="warning",
                        experiment=group_key,
                        message=f"Repeated pressures in '{group_key}' will be averaged deterministically for fitting; all source observations are retained.",
                    )
                )
            elif repeated:
                issues.append(
                    ImportIssue(
                        code="duplicates_kept",
                        severity="warning",
                        experiment=group_key,
                        message=f"Repeated pressures in '{group_key}' were retained as replicates.",
                    )
                )

            if len(observations) < 2:
                issues.append(
                    ImportIssue(
                        code="insufficient_observations",
                        severity="error",
                        experiment=group_key,
                        message="Each experiment requires at least two valid observations.",
                    )
                )
            for sequence_index, observation in enumerate(observations):
                observation["sequence_index"] = sequence_index
            experiments.append(
                {
                    "external_key": group_key,
                    "name": metadata["name"],
                    "adsorbent": {"name": metadata["adsorbent"]},
                    "adsorbates": [
                        {
                            "name": metadata["adsorbate"],
                            "molar_mass_g_mol": metadata.get(
                                "adsorbate_molar_mass_g_mol"
                            ),
                            "smiles": metadata.get("adsorbate_smiles"),
                        }
                    ],
                    "temperature_original": metadata["temperature_original"],
                    "temperature_original_unit": metadata["temperature_original_unit"],
                    "temperature_k": metadata["temperature_k"],
                    "pressure_basis": metadata["pressure_basis"],
                    "duplicate_policy": mapping.duplicate_policy,
                    "saturation_pressure_pa": metadata.get("saturation_pressure_pa"),
                    "conditions": metadata.get("conditions", {}),
                    "provenance": {"source_rows": metadata.get("source_rows", [])},
                    "observations": observations,
                }
            )

        return self._bundle(mapping, payload, experiments, issues)

    # -------------------------------------------------------------------------
    @staticmethod
    def _mapped_value(
        row: pd.Series,
        role: str,
        roles: dict[str, str],
        mapping: ImportMapping,
    ) -> Any:
        column = roles.get(role)
        if column is not None:
            return row.get(column)
        return mapping.constants.get(role)

    # -------------------------------------------------------------------------
    def _normalize_row(
        self,
        row: pd.Series,
        source_row: int,
        roles: dict[str, str],
        mapping: ImportMapping,
        detected: dict[str, ColumnDetection],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        adsorbent = str(
            self._mapped_value(row, "adsorbent", roles, mapping) or ""
        ).strip()
        adsorbate = str(
            self._mapped_value(row, "adsorbate", roles, mapping) or ""
        ).strip()
        adsorbate_smiles_value = safe_cell(
            self._mapped_value(row, "adsorbate_smiles", roles, mapping)
        )
        adsorbate_smiles = (
            str(adsorbate_smiles_value).strip()
            if adsorbate_smiles_value is not None
            else None
        )
        if adsorbate_smiles and adsorbate_smiles.casefold() in {"nan", "none"}:
            adsorbate_smiles = None
        if not adsorbent or adsorbent.casefold() in {"nan", "none"}:
            raise ValueError("Adsorbent material is missing.")
        if not adsorbate or adsorbate.casefold() in {"nan", "none"}:
            raise ValueError("Adsorbate species is missing.")

        temperature_value = parse_number(
            self._mapped_value(row, "temperature", roles, mapping),
            mapping.decimal_separator,
        )
        temperature_unit = self._unit_for("temperature", row, roles, mapping, detected)
        temperature = UnitRegistry.convert_temperature(
            temperature_value, temperature_unit
        )

        pressure_unit = self._unit_for("pressure", row, roles, mapping, detected)
        uptake_unit = self._unit_for("uptake", row, roles, mapping, detected)
        known_molar_masses = {
            "co2": 44.0095,
            "carbon dioxide": 44.0095,
            "n2": 28.0134,
            "nitrogen": 28.0134,
            "ch4": 16.0425,
            "methane": 16.0425,
            "o2": 31.9988,
            "h2": 2.01588,
            "hydrogen": 2.01588,
        }
        molar_mass = mapping.constants.get("adsorbate_molar_mass_g_mol")
        if molar_mass is None:
            molar_mass = known_molar_masses.get(adsorbate.casefold())
        molar_mass_value = float(molar_mass) if molar_mass not in (None, "") else None
        name = str(
            self._mapped_value(row, "experiment_name", roles, mapping) or ""
        ).strip()
        if not name or name.casefold() in {"nan", "none"}:
            name = " | ".join(
                str(row.get(column, "")).strip() for column in mapping.grouping_columns
            )

        saturation_pressure_pa = None
        if self._mapped_value(row, "saturation_pressure", roles, mapping) not in (
            None,
            "",
        ):
            saturation_value = parse_number(
                self._mapped_value(row, "saturation_pressure", roles, mapping),
                mapping.decimal_separator,
            )
            saturation_unit = mapping.unit_overrides.get(
                "saturation_pressure", pressure_unit
            )
            saturation_pressure_pa = UnitRegistry.convert_pressure(
                saturation_value, saturation_unit, "absolute"
            ).canonical_value

        pairs: list[tuple[float, float, int]] = []
        pressure_raw = self._mapped_value(row, "pressure", roles, mapping)
        uptake_raw = self._mapped_value(row, "uptake", roles, mapping)
        if mapping.wide_pairs:
            for pair_index, pair in enumerate(mapping.wide_pairs):
                pressure = parse_number(
                    row.get(pair.pressure_column), mapping.decimal_separator
                )
                uptake = parse_number(
                    row.get(pair.uptake_column), mapping.decimal_separator
                )
                pairs.append((pressure, uptake, pair_index))
        else:
            pressure_values = parse_series_cell(
                pressure_raw,
                delimiter=mapping.series_delimiter,
                decimal_separator=mapping.decimal_separator,
            )
            uptake_values = parse_series_cell(
                uptake_raw,
                delimiter=mapping.series_delimiter,
                decimal_separator=mapping.decimal_separator,
            )
            if len(pressure_values) != len(uptake_values):
                raise ValueError(
                    "Pressure and uptake series have unequal lengths "
                    f"({len(pressure_values)} and {len(uptake_values)})."
                )
            pairs = [
                (pressure, uptake, index)
                for index, (pressure, uptake) in enumerate(
                    zip(pressure_values, uptake_values, strict=True)
                )
            ]

        metadata_columns = [
            column
            for column, role in mapping.column_roles.items()
            if role == "metadata"
        ]
        observations: list[dict[str, Any]] = []
        for pressure_value, uptake_value, sequence_index in pairs:
            pressure = UnitRegistry.convert_pressure(
                pressure_value, pressure_unit, mapping.pressure_basis
            )
            uptake = UnitRegistry.convert_uptake(
                uptake_value, uptake_unit, molar_mass_value
            )
            stddev = None
            if self._mapped_value(row, "uptake_stddev", roles, mapping) not in (
                None,
                "",
            ):
                stddev_value = parse_number(
                    self._mapped_value(row, "uptake_stddev", roles, mapping),
                    mapping.decimal_separator,
                )
                stddev = UnitRegistry.convert_uptake(
                    stddev_value, uptake_unit, molar_mass_value
                ).canonical_value
                if stddev <= 0:
                    raise ValueError("Uptake uncertainty must be positive.")
            observations.append(
                {
                    "adsorbate": adsorbate,
                    "sequence_index": sequence_index,
                    "source_row": source_row,
                    "pressure_original": pressure.original_value,
                    "pressure_original_unit": pressure.original_unit,
                    "pressure_canonical": pressure.canonical_value,
                    "pressure_canonical_unit": pressure.canonical_unit,
                    "uptake_original": uptake.original_value,
                    "uptake_original_unit": uptake.original_unit,
                    "uptake_mol_kg": uptake.canonical_value,
                    "uptake_stddev_mol_kg": stddev,
                    "conversion_metadata": {
                        "pressure_rule": pressure.rule,
                        "uptake_rule": uptake.rule,
                    },
                    "extra_metadata": {
                        "source_pressure_token": safe_cell(pressure_raw),
                        "source_uptake_token": safe_cell(uptake_raw),
                        **{
                            column: safe_cell(row.get(column))
                            for column in metadata_columns
                        },
                    },
                }
            )
        return observations, {
            "name": name,
            "adsorbent": adsorbent,
            "adsorbate": adsorbate,
            "adsorbate_smiles": adsorbate_smiles,
            "adsorbate_molar_mass_g_mol": molar_mass_value,
            "temperature_original": temperature.original_value,
            "temperature_original_unit": temperature.original_unit,
            "temperature_k": temperature.canonical_value,
            "pressure_basis": mapping.pressure_basis,
            "saturation_pressure_pa": saturation_pressure_pa,
            "source_rows": [source_row],
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def _unit_for(
        quantity: str,
        row: pd.Series,
        roles: dict[str, str],
        mapping: ImportMapping,
        detected: dict[str, ColumnDetection],
    ) -> str:
        override = mapping.unit_overrides.get(quantity)
        if override:
            return override
        unit_role = f"{quantity}_unit"
        unit_column = roles.get(unit_role)
        if unit_column:
            value = row.get(unit_column)
            if value is not None and str(value).strip():
                return str(value).strip()
            raise UnitConversionError(
                f"The {quantity.replace('_', ' ')} unit column is empty for this observation."
            )
        value_column = roles.get(quantity)
        if value_column and detected[value_column].detected_unit:
            return str(detected[value_column].detected_unit)
        constant = mapping.constants.get(unit_role)
        if constant:
            return str(constant)
        raise UnitConversionError(
            f"The {quantity.replace('_', ' ')} unit is unknown; choose it explicitly."
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _average_duplicates(
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[float, list[dict[str, Any]]] = {}
        for item in observations:
            grouped.setdefault(item["pressure_canonical"], []).append(item)
        averaged: list[dict[str, Any]] = []
        for pressure, items in sorted(grouped.items()):
            if len(items) == 1:
                averaged.append(items[0])
                continue
            base = dict(items[0])
            base["uptake_mol_kg"] = sum(item["uptake_mol_kg"] for item in items) / len(
                items
            )
            base["uptake_original"] = base["uptake_mol_kg"]
            base["uptake_original_unit"] = "mol/kg"
            base["conversion_metadata"] = {
                **base["conversion_metadata"],
                "duplicate_policy": "average",
                "replicate_count": len(items),
                "source_uptake_values_mol_kg": [
                    item["uptake_mol_kg"] for item in items
                ],
            }
            averaged.append(base)
        return averaged

    # -------------------------------------------------------------------------
    @staticmethod
    def _bundle(
        mapping: ImportMapping,
        payload: bytes,
        experiments: list[dict[str, Any]],
        issues: list[ImportIssue],
    ) -> ValidationBundle:
        errors = [issue for issue in issues if issue.severity == "error"]
        confirmations = [
            issue
            for issue in issues
            if issue.severity == "confirmation"
            and issue.code not in mapping.confirmed_issue_codes
        ]
        status = (
            "invalid"
            if errors
            else "confirmation_required"
            if confirmations
            else "valid"
        )
        previews = [
            NormalizedExperimentPreview(
                external_key=experiment["external_key"],
                name=experiment["name"],
                adsorbent=experiment["adsorbent"]["name"],
                adsorbate=experiment["adsorbates"][0]["name"],
                adsorbate_smiles=experiment["adsorbates"][0].get("smiles"),
                temperature_k=experiment["temperature_k"],
                pressure_basis=experiment["pressure_basis"],
                observation_count=len(experiment["observations"]),
                observations=[
                    NormalizedObservationPreview(
                        **{
                            key: observation[key]
                            for key in (
                                "source_row",
                                "sequence_index",
                                "pressure_original",
                                "pressure_original_unit",
                                "pressure_canonical",
                                "pressure_canonical_unit",
                                "uptake_original",
                                "uptake_original_unit",
                                "uptake_mol_kg",
                            )
                        }
                    )
                    for observation in experiment["observations"][
                        :EXPERIMENT_PREVIEW_POINTS
                    ]
                ],
            )
            for experiment in experiments[:20]
        ]
        response = ImportValidationResponse(
            status=status,  # type: ignore[arg-type]
            source_sha256=source_hash(payload),
            structure=mapping.structure,
            experiment_count=len(experiments),
            observation_count=sum(
                len(experiment["observations"]) for experiment in experiments
            ),
            experiments=previews,
            issues=issues,
        )
        return ValidationBundle(response=response, experiments=experiments)
