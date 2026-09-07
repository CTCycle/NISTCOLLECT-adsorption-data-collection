from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal


PressureBasis = Literal["absolute", "partial", "relative"]

###############################################################################
class UnitConversionError(ValueError):
    pass

###############################################################################
@dataclass(frozen=True)
class ConvertedValue:
    original_value: float
    original_unit: str
    canonical_value: float
    canonical_unit: str
    rule: str

###############################################################################
def normalize_token(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = text.replace("−", "-").replace("·", " ").replace("³", "3")
    return " ".join(text.split())

###############################################################################
def parse_number(value: Any, decimal_separator: str = "auto") -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid measurements.")
    if isinstance(value, (int, float)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Measurement must be finite.")
        return result

    text = normalize_token(value).replace("\u00a0", " ").strip()
    if not text:
        raise ValueError("Measurement is empty.")
    match = re.fullmatch(
        r"\s*([+-]?(?:\d[\d\s.,']*|\d*[.,]\d+)(?:[eE][+-]?\d+)?)\s*(?:[^\d].*)?",
        text,
    )
    if match is None:
        raise ValueError(f"'{value}' is not a valid numerical value.")
    numeric = match.group(1).replace(" ", "").replace("'", "")

    if decimal_separator not in {"auto", ".", ","}:
        raise ValueError("Decimal separator must be 'auto', '.', or ','.")
    if decimal_separator == ".":
        numeric = numeric.replace(",", "")
    elif decimal_separator == ",":
        numeric = numeric.replace(".", "").replace(",", ".")
    elif "." in numeric and "," in numeric:
        decimal = "." if numeric.rfind(".") > numeric.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        numeric = numeric.replace(thousands, "").replace(decimal, ".")
    elif "," in numeric:
        if numeric.count(",") == 1 and len(numeric.rsplit(",", 1)[1]) == 3:
            raise ValueError(
                f"'{value}' is ambiguous; select comma-decimal or comma-thousands explicitly."
            )
        numeric = numeric.replace(",", ".")
    elif (
        "." in numeric
        and numeric.count(".") == 1
        and len(numeric.rsplit(".", 1)[1]) == 3
    ):
        raise ValueError(
            f"'{value}' is ambiguous; select decimal-point or thousands-point explicitly."
        )

    try:
        result = float(numeric)
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a valid numerical value.") from exc
    if not math.isfinite(result):
        raise ValueError("Measurement must be finite.")
    return result

###############################################################################
class UnitRegistry:
    GAS_CONSTANT_J_MOL_K = 8.31446261815324
    STP_MOLAR_VOLUME_L_MOL = 22.41396954

    PRESSURE_ALIASES = {
        "pa": "Pa",
        "pascal": "Pa",
        "pascals": "Pa",
        "kpa": "kPa",
        "kilopascal": "kPa",
        "mpa": "MPa",
        "megapascal": "MPa",
        "bar": "bar",
        "mbar": "mbar",
        "atm": "atm",
        "atmosphere": "atm",
        "torr": "torr",
        "mmhg": "torr",
        "psi": "psi",
        "p/p0": "1",
        "p/p₀": "1",
        "relative": "1",
        "relative pressure": "1",
        "1": "1",
        "%": "%",
        "percent": "%",
    }
    PRESSURE_TO_PA = {
        "Pa": 1.0,
        "kPa": 1_000.0,
        "MPa": 1_000_000.0,
        "bar": 100_000.0,
        "mbar": 100.0,
        "atm": 101_325.0,
        "torr": 133.32236842105263,
        "psi": 6_894.757293168,
    }
    UPTAKE_ALIASES = {
        "mol/kg": "mol/kg",
        "mol kg-1": "mol/kg",
        "mol kg^-1": "mol/kg",
        "mmol/g": "mmol/g",
        "mmol g-1": "mmol/g",
        "mmol g^-1": "mmol/g",
        "mmol/kg": "mmol/kg",
        "mol/g": "mol/g",
        "mg/g": "mg/g",
        "g/g": "g/g",
        "wt%": "wt%",
        "weight %": "wt%",
        "g/100g": "wt%",
        "g adsorbate / 100g adsorbent": "wt%",
        "cm3(stp)/g": "cm3(STP)/g",
        "cm3 stp/g": "cm3(STP)/g",
        "cm3/g stp": "cm3(STP)/g",
        "ml(stp)/g": "cm3(STP)/g",
        "ml stp/g": "cm3(STP)/g",
    }
    TEMPERATURE_ALIASES = {
        "k": "K",
        "kelvin": "K",
        "°c": "degC",
        "c": "degC",
        "degc": "degC",
        "celsius": "degC",
        "°f": "degF",
        "f": "degF",
        "degf": "degF",
        "fahrenheit": "degF",
    }

    # -------------------------------------------------------------------------
    @classmethod
    def pressure_unit(cls, unit: object) -> str:
        normalized = normalize_token(unit).replace(" ", "")
        aliases = {
            key.replace(" ", ""): value for key, value in cls.PRESSURE_ALIASES.items()
        }
        resolved = aliases.get(normalized)
        if resolved is None:
            raise UnitConversionError(f"Unsupported pressure unit '{unit}'.")
        return resolved

    # -------------------------------------------------------------------------
    @classmethod
    def uptake_unit(cls, unit: object) -> str:
        normalized = normalize_token(unit).replace(" ", "")
        aliases = {
            key.replace(" ", ""): value for key, value in cls.UPTAKE_ALIASES.items()
        }
        resolved = aliases.get(normalized)
        if resolved is None:
            raise UnitConversionError(f"Unsupported uptake unit '{unit}'.")
        return resolved

    # -------------------------------------------------------------------------
    @classmethod
    def temperature_unit(cls, unit: object) -> str:
        normalized = normalize_token(unit).replace(" ", "")
        aliases = {
            key.replace(" ", ""): value
            for key, value in cls.TEMPERATURE_ALIASES.items()
        }
        resolved = aliases.get(normalized)
        if resolved is None:
            raise UnitConversionError(f"Unsupported temperature unit '{unit}'.")
        return resolved

    # -------------------------------------------------------------------------
    @classmethod
    def convert_pressure(
        cls, value: float, unit: object, basis: PressureBasis
    ) -> ConvertedValue:
        resolved = cls.pressure_unit(unit)
        if basis == "relative":
            if resolved == "1":
                canonical = value
                rule = "dimensionless identity"
            elif resolved == "%":
                canonical = value / 100.0
                rule = "percent / 100"
            else:
                raise UnitConversionError(
                    "Relative pressure requires p/p0, dimensionless, or percent units."
                )
            if not 0 <= canonical <= 1:
                raise UnitConversionError("Relative pressure must be between 0 and 1.")
            return ConvertedValue(value, resolved, canonical, "1", rule)

        if resolved not in cls.PRESSURE_TO_PA:
            raise UnitConversionError(
                f"{basis.capitalize()} pressure requires a dimensional pressure unit."
            )
        factor = cls.PRESSURE_TO_PA[resolved]
        canonical = value * factor
        if canonical < 0:
            raise UnitConversionError("Pressure must not be negative.")
        return ConvertedValue(
            value, resolved, canonical, "Pa", f"{resolved} * {factor:g}"
        )

    # -------------------------------------------------------------------------
    @classmethod
    def convert_temperature(cls, value: float, unit: object) -> ConvertedValue:
        resolved = cls.temperature_unit(unit)
        if resolved == "K":
            canonical = value
            rule = "kelvin identity"
        elif resolved == "degC":
            canonical = value + 273.15
            rule = "degC + 273.15"
        else:
            canonical = (value - 32.0) * 5.0 / 9.0 + 273.15
            rule = "(degF - 32) * 5/9 + 273.15"
        if canonical <= 0:
            raise UnitConversionError("Temperature must be above absolute zero.")
        return ConvertedValue(value, resolved, canonical, "K", rule)

    # -------------------------------------------------------------------------
    @classmethod
    def uptake_factor_to_mol_kg(
        cls, unit: object, molar_mass_g_mol: float | None = None
    ) -> tuple[str, float, str]:
        resolved = cls.uptake_unit(unit)
        if resolved in {"mol/kg", "mmol/g"}:
            return resolved, 1.0, f"{resolved} == mol/kg"
        if resolved == "mmol/kg":
            return resolved, 0.001, "mmol/kg / 1000"
        if resolved == "mol/g":
            return resolved, 1000.0, "mol/g * 1000"
        if resolved == "cm3(STP)/g":
            factor = 1.0 / cls.STP_MOLAR_VOLUME_L_MOL
            return (
                resolved,
                factor,
                f"cm3(STP)/g / {cls.STP_MOLAR_VOLUME_L_MOL:g} L/mol",
            )
        if molar_mass_g_mol is None or molar_mass_g_mol <= 0:
            raise UnitConversionError(
                f"Uptake unit '{resolved}' requires a positive adsorbate molar mass."
            )
        if resolved == "mg/g":
            return resolved, 1.0 / molar_mass_g_mol, "mg/g / molar mass"
        if resolved == "g/g":
            return resolved, 1000.0 / molar_mass_g_mol, "g/g * 1000 / molar mass"
        if resolved == "wt%":
            return resolved, 10.0 / molar_mass_g_mol, "wt% * 10 / molar mass"
        raise UnitConversionError(f"Unsupported uptake unit '{unit}'.")

    # -------------------------------------------------------------------------
    @classmethod
    def convert_uptake(
        cls, value: float, unit: object, molar_mass_g_mol: float | None = None
    ) -> ConvertedValue:
        resolved, factor, rule = cls.uptake_factor_to_mol_kg(unit, molar_mass_g_mol)
        canonical = value * factor
        if canonical < 0:
            raise UnitConversionError("Adsorption uptake must not be negative.")
        return ConvertedValue(value, resolved, canonical, "mol/kg", rule)

    # -------------------------------------------------------------------------
    @classmethod
    def pressure_from_pa(cls, value_pa: float, unit: object) -> float:
        resolved = cls.pressure_unit(unit)
        if resolved not in cls.PRESSURE_TO_PA:
            raise UnitConversionError(
                "A dimensional display pressure unit is required."
            )
        return value_pa / cls.PRESSURE_TO_PA[resolved]

    # -------------------------------------------------------------------------
    @classmethod
    def uptake_from_mol_kg(
        cls, value: float, unit: object, molar_mass_g_mol: float | None = None
    ) -> float:
        _, factor, _ = cls.uptake_factor_to_mol_kg(unit, molar_mass_g_mol)
        return value / factor


HEADER_UNIT_PATTERN = re.compile(r"(?:\[|\()([^\])]+)(?:\]|\))")

###############################################################################
def detect_header_unit(header: str, quantity: str) -> str | None:
    candidates = [match.strip() for match in HEADER_UNIT_PATTERN.findall(header)]
    candidates.extend(
        token
        for token in (
            "p/p0",
            "bar",
            "kpa",
            "mpa",
            "pa",
            "mmol/g",
            "mol/kg",
            "mg/g",
            "cm3(stp)/g",
            "°c",
            "°f",
            "kelvin",
        )
        if token in normalize_token(header)
    )
    resolver = {
        "pressure": UnitRegistry.pressure_unit,
        "uptake": UnitRegistry.uptake_unit,
        "temperature": UnitRegistry.temperature_unit,
    }[quantity]
    for candidate in candidates:
        try:
            return resolver(candidate)
        except UnitConversionError:
            continue
    return None
