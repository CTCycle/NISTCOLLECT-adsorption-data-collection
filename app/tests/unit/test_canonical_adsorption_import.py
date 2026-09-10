from __future__ import annotations

import pytest

from adsmod_core.contracts.datasets import ImportMapping
from adsmod_core.services.data.importer import AdsorptionImportEngine

###############################################################################
def test_atomic_import_groups_rows_and_normalizes_units() -> None:
    payload = b"experiment_id,Pressure [bar],Uptake [mmol/g],Temperature [K],Adsorbate,Adsorbent\nEXP-1,0.1,0.42,298.15,CO2,13X\nEXP-1,0.2,0.73,298.15,CO2,13X\n"
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "sample.csv")
    mapping = ImportMapping(
        dataset_name="sample",
        structure="atomic",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
    )
    bundle = engine.validate(payload, "sample.csv", mapping)
    assert bundle.response.status == "valid"
    assert bundle.response.experiment_count == 1
    assert bundle.response.observation_count == 2
    assert bundle.experiments[0]["observations"][0]["pressure_canonical"] == 10_000
    assert bundle.experiments[0]["observations"][0]["uptake_mol_kg"] == 0.42

###############################################################################
def test_aggregated_arrays_require_equal_lengths() -> None:
    payload = b'experiment_id,pressure,uptake,temperature,adsorbate,adsorbent\nEXP-1,"[1;2]","[3]",298.15,CO2,13X\n'
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "sample.csv")
    mapping = ImportMapping(
        dataset_name="sample",
        structure="aggregated",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
        series_delimiter=";",
        unit_overrides={"pressure": "bar", "uptake": "mmol/g", "temperature": "K"},
    )
    bundle = engine.validate(payload, "sample.csv", mapping)
    assert bundle.response.status == "invalid"
    assert any(issue.code == "invalid_row" for issue in bundle.response.issues)

###############################################################################
def test_atomic_import_preserves_adsorbate_smiles_for_training() -> None:
    payload = b"experiment_id,pressure [kPa],uptake [mol/g],temperature [K],adsorbate,adsorbate_smiles,adsorbent\nEXP-1,10,0.10,298.15,CO2,O=C=O,Activated carbon\nEXP-1,20,0.18,298.15,CO2,O=C=O,Activated carbon\n"
    engine = AdsorptionImportEngine()
    preview = engine.preview(payload, "smiles.csv")
    mapping = ImportMapping(
        dataset_name="smiles",
        structure="atomic",
        column_roles={column.name: column.proposed_role for column in preview.columns},
        grouping_columns=["experiment_id"],
        pressure_basis="absolute",
    )

    bundle = engine.validate(payload, "smiles.csv", mapping)

    assert bundle.response.status == "valid"
    assert bundle.experiments[0]["adsorbates"][0]["smiles"] == "O=C=O"
    assert bundle.response.experiments[0].adsorbate_smiles == "O=C=O"

###############################################################################
def test_import_rejects_extensions_outside_the_canonical_policy() -> None:
    payload = b"pressure,uptake,temperature,adsorbate,adsorbent\n1,2,298,CO2,13X\n"

    with pytest.raises(ValueError, match=r"Allowed file types: \.csv, \.xls, \.xlsx"):
        AdsorptionImportEngine().preview(payload, "sample.json")


###############################################################################
def test_import_engine_accepts_extensions_supplied_by_runtime_configuration() -> None:
    preview = AdsorptionImportEngine(allowed_extensions=(".json",)).preview(
        b'[{"pressure": 1, "uptake": 2, "temperature": 298, "adsorbate": "CO2", "adsorbent": "13X"}]',
        "sample.json",
    )

    assert preview.filename == "sample.json"
