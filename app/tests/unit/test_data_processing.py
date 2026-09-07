from __future__ import annotations

import pandas as pd
import pytest
from pandas.api.types import is_string_dtype

from adsmod_common.units import UnitRegistry
from adsmod_core.common.utils.encoding import sanitize_dataframe_strings
from adsmod_ml.services.data.conversion import PressureConversion, UptakeConversion
from adsmod_ml.services.data.sanitizer import DataSanitizer

###############################################################################
def test_sanitize_dataframe_strings_handles_pandas_string_dtype() -> None:
    frame = pd.DataFrame({"name": ["zeolite\u200b-a", "na\u00a0y"]}).astype("string")

    sanitized = sanitize_dataframe_strings(frame)

    assert is_string_dtype(sanitized["name"])
    assert sanitized.loc[0, "name"] == "zeolite-a"
    assert sanitized.loc[1, "name"] == "na y"

###############################################################################
def test_pressure_conversion_returns_frame_without_unit_column() -> None:
    converter = PressureConversion()
    frame = pd.DataFrame({"pressure": [1.0], "pressure_units": ["bar"]})

    converted = converter.convert_pressure_units(frame)

    assert "pressure_units" not in converted.columns
    assert converted.loc[0, "pressure"] == 100000.0

###############################################################################
@pytest.mark.parametrize(
    "unit", ["Pa", "kPa", "MPa", "bar", "mbar", "atm", "torr", "mmhg", "psi"]
)
def test_pressure_conversion_uses_canonical_registry(unit: str) -> None:
    resolved = UnitRegistry.pressure_unit(unit)
    frame = pd.DataFrame({"pressure": [2.0], "pressure_units": [unit]})

    converted = PressureConversion().convert_pressure_units(frame)

    assert converted.loc[0, "pressure"] == 2.0 * UnitRegistry.PRESSURE_TO_PA[resolved]

###############################################################################
@pytest.mark.parametrize(
    "unit",
    [
        "mmol/g",
        "mol/kg",
        "mmol/kg",
        "mol/g",
        "mg/g",
        "g/g",
        "wt%",
        "g/100g",
        "g adsorbate / 100g adsorbent",
        "cm3(stp)/g",
        "ml(stp)/g",
    ],
)
def test_uptake_conversion_uses_canonical_registry(unit: str) -> None:
    frame = pd.DataFrame(
        {
            "adsorbed_amount": [2.0],
            "adsorption_units": [unit],
            "adsorbate_molecular_weight": [18.0],
        }
    )

    converted = UptakeConversion().convert_uptake_data(frame)
    expected = UnitRegistry.convert_uptake(2.0, unit, 18.0).canonical_value

    assert converted.loc[0, "adsorbed_amount"] == pytest.approx(expected)
    assert "adsorption_units" not in converted.columns

###############################################################################
def test_exclude_oob_values_uses_copy_safe_assignment() -> None:
    sanitizer = DataSanitizer({"max_pressure": 10, "max_uptake": 20})
    frame = pd.DataFrame(
        {
            "temperature": [300, -1],
            "pressure": [[0.0, 12_000_000.0, 5.0], [1.0]],
            "adsorbed_amount": [[1.0, 2.0, 30.0], [1.0]],
        }
    )

    filtered = sanitizer.exclude_OOB_values(frame)

    assert len(filtered) == 1
    assert filtered.iloc[0]["pressure"] == [0.0]
    assert filtered.iloc[0]["adsorbed_amount"] == [1.0]
    assert frame.loc[0, "pressure"] == [0.0, 12_000_000.0, 5.0]
