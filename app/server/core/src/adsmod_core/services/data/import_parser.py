from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import re
import unicodedata
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

from adsmod_common.config import DEFAULT_DATASET_ALLOWED_EXTENSIONS
from adsmod_core.contracts.datasets import ColumnDetection
from adsmod_common.units import detect_header_unit, parse_number


ROLE_ALIASES: dict[str, set[str]] = {
    "experiment_id": {
        "experiment",
        "experiment id",
        "experiment_id",
        "isotherm",
        "isotherm id",
        "isotherm_id",
        "exp id",
        "id esperimento",
        "esperimento",
        "id experimento",
        "experimento",
        "id experience",
        "experience",
        "versuch id",
        "versuch",
        "experiencia id",
    },
    "experiment_name": {
        "experiment name",
        "experiment_name",
        "isotherm name",
        "nome esperimento",
        "nombre experimento",
        "nom experience",
        "versuchsname",
        "nome experimento",
    },
    "pressure": {
        "pressure",
        "equilibrium pressure",
        "absolute pressure",
        "partial pressure",
        "relative pressure",
        "p",
        "peq",
        "p/p0",
        "pressione",
        "pressione equilibrio",
        "presion",
        "presion equilibrio",
        "pression",
        "pression equilibre",
        "druck",
        "gleichgewichtsdruck",
        "pressao",
    },
    "uptake": {
        "uptake",
        "adsorption",
        "adsorbed amount",
        "adsorbed quantity",
        "loading",
        "adsorption capacity",
        "capacity",
        "q",
        "qe",
        "quantita adsorbita",
        "carico",
        "cantidad adsorbida",
        "carga",
        "quantite adsorbee",
        "chargement",
        "beladung",
        "adsorbierte menge",
        "quantidade adsorvida",
    },
    "adsorbate": {
        "adsorbate",
        "adsorbate species",
        "species",
        "gas",
        "guest",
        "sorbate",
        "adsorbato",
        "gas adsorbito",
        "especie",
        "adsorbat",
        "gast",
        "adsorvato",
    },
    "adsorbate_smiles": {
        "smile",
        "smiles",
        "smile code",
        "smile_code",
        "adsorbate smile",
        "adsorbate smiles",
        "adsorbate smile code",
        "canonical smile",
        "canonical smiles",
    },
    "adsorbent": {
        "adsorbent",
        "material",
        "adsorbent material",
        "host",
        "sorbent",
        "adsorbente",
        "materiale",
        "materiau",
        "adsorbens",
        "werkstoff",
        "material adsorvente",
    },
    "temperature": {
        "temperature",
        "temp",
        "t",
        "temperatura",
        "temperature equilibrium",
        "temperatur",
    },
    "pressure_unit": {
        "pressure unit",
        "pressure units",
        "unita pressione",
        "unidad presion",
        "unite pression",
        "druckeinheit",
    },
    "uptake_unit": {
        "uptake unit",
        "adsorption unit",
        "adsorption units",
        "unita adsorbimento",
        "unidad adsorcion",
        "unite adsorption",
        "beladungseinheit",
    },
    "temperature_unit": {
        "temperature unit",
        "unita temperatura",
        "unidad temperatura",
        "unite temperature",
        "temperatureinheit",
    },
    "uptake_stddev": {
        "uptake stddev",
        "uptake standard deviation",
        "adsorption uncertainty",
        "sigma q",
        "q std",
    },
    "saturation_pressure": {
        "saturation pressure",
        "vapor pressure",
        "vapour pressure",
        "p0",
        "pressione saturazione",
        "presion saturacion",
        "pression saturation",
        "sattigungsdruck",
    },
}

###############################################################################
def normalize_header(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[\[\](){}]", " ", text)
    text = re.sub(r"[^a-z0-9/°%]+", " ", text)
    return " ".join(text.split())

###############################################################################
def safe_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

###############################################################################
def source_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

###############################################################################
def read_tabular(
    payload: bytes,
    filename: str | None,
    *,
    allowed_extensions: Collection[str] = DEFAULT_DATASET_ALLOWED_EXTENSIONS,
    header_row: int = 0,
    field_delimiter: str | None = None,
    encoding: str = "utf-8",
    worksheet: str | int | None = None,
) -> pd.DataFrame:
    if not payload:
        raise ValueError("Uploaded dataset is empty.")
    suffix = Path(filename or "").suffix.casefold()
    normalized_extensions = {
        extension.strip().casefold()
        if extension.strip().startswith(".")
        else f".{extension.strip().casefold()}"
        for extension in allowed_extensions
        if extension.strip()
    }
    if suffix not in normalized_extensions:
        allowed = ", ".join(sorted(normalized_extensions))
        raise ValueError(
            f"Unsupported file type '{suffix or 'unknown'}'. Allowed file types: {allowed}."
        )
    buffer = io.BytesIO(payload)
    try:
        if suffix in {".xls", ".xlsx"}:
            frame = pd.read_excel(
                buffer,
                sheet_name=worksheet if worksheet is not None else 0,
                header=header_row,
                dtype=object,
            )
        elif suffix == ".json":
            decoded = json.loads(payload.decode(encoding if encoding else "utf-8-sig"))
            if isinstance(decoded, dict):
                decoded = decoded.get("records", decoded.get("data"))
            if not isinstance(decoded, list) or not all(
                isinstance(row, dict) for row in decoded
            ):
                raise ValueError("JSON datasets must contain an array of row objects.")
            frame = pd.DataFrame.from_records(decoded)
        elif suffix in {".csv", ".txt", ".tsv", ""}:
            frame = pd.read_csv(
                io.StringIO(payload.decode(encoding)),
                sep=field_delimiter or None,
                engine="python",
                header=header_row,
                dtype=object,
                comment="#",
                keep_default_na=True,
            )
        else:
            raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'.")
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to parse '{filename or 'dataset'}': {exc}") from exc
    if frame.empty:
        raise ValueError("Uploaded dataset contains no data rows.")
    if len(frame.columns) > 256:
        raise ValueError("Uploaded dataset contains more than 256 columns.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise ValueError("Every uploaded column must have a non-empty header.")
    if len(set(frame.columns)) != len(frame.columns):
        raise ValueError("Uploaded dataset contains duplicate column headers.")
    return frame

###############################################################################
def parse_series_cell(
    value: Any,
    *,
    delimiter: str | None,
    decimal_separator: str,
) -> list[float]:
    if isinstance(value, (list, tuple)):
        raw = list(value)
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    elif isinstance(value, (int, float)):
        return [parse_number(value, decimal_separator)]
    else:
        text = str(value).strip()
        if not text:
            return []
        raw: Any = None
        if text.startswith("[") and text.endswith("]"):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                try:
                    raw = ast.literal_eval(text)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(
                        f"Malformed serialized array '{text[:80]}'."
                    ) from exc
            if not isinstance(raw, (list, tuple)):
                raise ValueError("Serialized measurement cells must contain an array.")
            raw = list(raw)
        else:
            selected = delimiter
            if selected is None:
                candidates = [token for token in (";", "|", "\t") if token in text]
                if not candidates:
                    return [parse_number(text, decimal_separator)]
                if len(candidates) != 1:
                    raise ValueError(
                        "Delimited series require an explicit, unambiguous delimiter."
                    )
                selected = candidates[0]
            raw = [item.strip() for item in text.split(selected)]
    return [parse_number(item, decimal_separator) for item in raw]

###############################################################################
def is_array_like(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    return (text.startswith("[") and text.endswith("]")) or sum(
        token in text for token in (";", "|", "\t")
    ) == 1

###############################################################################
def infer_column(column: str, series: pd.Series) -> ColumnDetection:
    header = normalize_header(column)
    non_empty = [safe_cell(value) for value in series.dropna().head(20).tolist()]
    array_ratio = (
        sum(is_array_like(value) for value in non_empty) / len(non_empty)
        if non_empty
        else 0.0
    )
    numeric_count = 0
    for value in non_empty:
        try:
            parse_number(value, ".")
            numeric_count += 1
        except ValueError:
            pass
    numeric_ratio = numeric_count / len(non_empty) if non_empty else 0.0

    scored: list[tuple[float, str, list[str]]] = []
    header_tokens = set(header.split())
    for role, aliases in ROLE_ALIASES.items():
        best = 0.0
        evidence: list[str] = []
        for alias in aliases:
            normalized_alias = normalize_header(alias)
            alias_tokens = set(normalized_alias.split())
            if header == normalized_alias:
                best = max(best, 0.98)
                evidence = [f"header exactly matches '{alias}'"]
            elif alias_tokens and alias_tokens.issubset(header_tokens):
                score = 0.82 if len(alias_tokens) > 1 else 0.72
                if score > best:
                    best = score
                    evidence = [f"header contains scientific alias '{alias}'"]
        if role in {"pressure", "uptake", "temperature", "uptake_stddev"}:
            if numeric_ratio >= 0.8:
                best += 0.08
                evidence.append("representative values are numerical")
            if array_ratio >= 0.5 and role in {"pressure", "uptake"}:
                best += 0.05
                evidence.append("representative values contain numerical series")
        if role == "experiment_id":
            cardinality = series.nunique(dropna=True)
            if 0 < cardinality < max(2, len(series)):
                best += 0.05
                evidence.append("values repeat and can define groups")
        if best:
            scored.append((min(best, 1.0), role, evidence))
    scored.sort(reverse=True)
    confidence, role, evidence = scored[0] if scored else (0.0, "ignore", [])

    detected_unit = None
    quantity = {
        "pressure": "pressure",
        "uptake": "uptake",
        "temperature": "temperature",
    }.get(role)
    if quantity:
        detected_unit = detect_header_unit(column, quantity)
        if detected_unit:
            confidence = min(1.0, confidence + 0.08)
            evidence.append(f"header declares unit '{detected_unit}'")

    inferred_type = (
        "series" if array_ratio >= 0.5 else "number" if numeric_ratio >= 0.8 else "text"
    )
    return ColumnDetection(
        name=column,
        inferred_type=inferred_type,
        sample_values=non_empty[:5],
        proposed_role=role,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        evidence=evidence,
        detected_unit=detected_unit,
        array_like=array_ratio >= 0.5,
    )

###############################################################################
def detect_structure(columns: list[ColumnDetection]) -> tuple[str, float]:
    pressure = next(
        (column for column in columns if column.proposed_role == "pressure"), None
    )
    uptake = next(
        (column for column in columns if column.proposed_role == "uptake"), None
    )
    wide_pressure = [
        column
        for column in columns
        if re.search(r"(?:pressure|\bp)\s*\d+$", normalize_header(column.name))
    ]
    wide_uptake = [
        column
        for column in columns
        if re.search(r"(?:uptake|\bq)\s*\d+$", normalize_header(column.name))
    ]
    if wide_pressure and len(wide_pressure) == len(wide_uptake):
        return "aggregated", 0.9
    if pressure and uptake:
        if pressure.array_like and uptake.array_like:
            return "aggregated", 0.95
        if pressure.array_like != uptake.array_like:
            return "mixed", 0.75
        return "atomic", 0.95
    return "ambiguous", 0.25

###############################################################################
def infer_pressure_basis(columns: list[ColumnDetection]) -> str | None:
    column = next((item for item in columns if item.proposed_role == "pressure"), None)
    if column is None:
        return None
    header = normalize_header(column.name)
    if "relative" in header or "p/p0" in header or column.detected_unit == "1":
        return "relative"
    if "partial" in header:
        return "partial"
    if "absolute" in header or column.detected_unit not in {None, "1", "%"}:
        return "absolute"
    return None
