from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from adsmod_ml.common.utils.logger import logger
from adsmod_common.units import UnitConversionError, UnitRegistry, normalize_token

###############################################################################
def map_values(
    values: list[int | float] | int | float | None,
    converter: Callable[[float], float],
) -> list[float] | float | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        converted: list[float] = []
        for value in values:
            if value is None or pd.isna(value):
                converted.append(float("nan"))
                continue
            converted.append(converter(float(value)))
        return converted
    if pd.isna(values):
        return None
    return converter(float(values))

###############################################################################
class PressureConversion:
    """Apply the canonical unit registry to pressure columns."""

    P_COL = "pressure"
    P_UNIT_COL = "pressure_units"

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_unit(unit: object) -> str:
        return normalize_token(unit)

    # -------------------------------------------------------------------------
    @staticmethod
    def convert_values(
        values: list[int | float] | int | float | None,
        unit: object,
    ) -> list[float] | float | None:
        try:
            resolved = UnitRegistry.pressure_unit(unit)
            factor = UnitRegistry.PRESSURE_TO_PA.get(resolved)
        except UnitConversionError:
            return values
        if factor is None:
            return values
        return map_values(values, lambda value: value * factor)

    # -------------------------------------------------------------------------
    def convert_pressure_units(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if (
            self.P_UNIT_COL not in dataframe.columns
            or self.P_COL not in dataframe.columns
        ):
            logger.debug("Pressure conversion skipped (missing pressure columns).")
            return dataframe

        dataframe[self.P_COL] = dataframe.apply(
            lambda row: self.convert_values(
                row.get(self.P_COL), row.get(self.P_UNIT_COL)
            ),
            axis=1,
        )
        return dataframe.drop(columns=self.P_UNIT_COL)

###############################################################################
class UptakeConversion:
    """Apply canonical uptake conversions while preserving the ML frame shape."""

    Q_COL = "adsorbed_amount"
    Q_UNIT_COL = "adsorption_units"
    MOLAR_MASS_COL = "adsorbate_molecular_weight"
    WEIGHT_UNITS = {"mg/g", "g/g", "wt%"}

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_unit(unit: object) -> str:
        return normalize_token(unit)

    # -------------------------------------------------------------------------
    def convert_uptake_row(
        self, row: pd.Series, unit_column: str
    ) -> list[float] | float | None:
        values = row.get(self.Q_COL)
        try:
            resolved = UnitRegistry.uptake_unit(row.get(unit_column))
        except UnitConversionError:
            return values

        molar_mass = row.get(self.MOLAR_MASS_COL)
        if resolved in self.WEIGHT_UNITS and (
            molar_mass in (None, 0, "") or pd.isna(molar_mass)
        ):
            return values

        def convert(value: float) -> float:
            return UnitRegistry.convert_uptake(
                value,
                resolved,
                None if molar_mass is None else float(molar_mass),
            ).canonical_value

        try:
            return map_values(values, convert)
        except (TypeError, ValueError, UnitConversionError):
            return values

    # -------------------------------------------------------------------------
    def convert_uptake_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if (
            self.Q_UNIT_COL not in dataframe.columns
            or self.Q_COL not in dataframe.columns
        ):
            logger.debug("Uptake conversion skipped (missing adsorption columns).")
            return dataframe

        dataframe[self.Q_COL] = [
            self.convert_uptake_row(row, self.Q_UNIT_COL)
            for _, row in dataframe.iterrows()
        ]
        return dataframe.drop(columns=self.Q_UNIT_COL)

###############################################################################
def PQ_units_conversion(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert pressure and uptake to canonical values and remove unit columns."""
    if dataframe.empty:
        return dataframe

    converted_data = PressureConversion().convert_pressure_units(dataframe)
    return UptakeConversion().convert_uptake_data(converted_data)
