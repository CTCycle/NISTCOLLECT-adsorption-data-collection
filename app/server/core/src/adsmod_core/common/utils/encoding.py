"""Unicode safety helpers for ingestion, persistence, and logging.

Policy:
- Treat inbound/outbound text as UTF-8.
- Preserve visible Unicode characters (scientific symbols, Greek letters, diacritics).
- Explicitly normalize only problematic invisible/control characters.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


SAFE_CONTROL_CHARACTERS = {"\n", "\r", "\t"}
ZERO_WIDTH_CHARACTERS = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\ufeff",  # ZERO WIDTH NO-BREAK SPACE / BOM
}

###############################################################################
def normalize_unicode_text(value: str) -> str:
    normalized = value.replace("\u00a0", " ")
    normalized = "".join(
        character for character in normalized if character not in ZERO_WIDTH_CHARACTERS
    )
    return "".join(
        character
        for character in normalized
        if ord(character) >= 32 or character in SAFE_CONTROL_CHARACTERS
    )

###############################################################################
def normalize_error_text(value: str) -> str:
    return "".join(
        character
        for character in normalize_unicode_text(value)
        if ord(character) < 128 or character in SAFE_CONTROL_CHARACTERS
    )

###############################################################################
def sanitize_unicode_payload(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_unicode_text(value)
    if isinstance(value, list):
        return [sanitize_unicode_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_unicode_payload(item) for item in value)
    if isinstance(value, dict):
        return {
            sanitize_unicode_payload(key): sanitize_unicode_payload(item)
            for key, item in value.items()
        }
    return value

###############################################################################
def sanitize_dataframe_strings(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    sanitized = dataframe.copy()
    string_columns = sanitized.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        sanitized[column] = sanitized[column].apply(
            lambda value: (
                normalize_unicode_text(value) if isinstance(value, str) else value
            )
        )
    return sanitized

###############################################################################
def decode_json_response_bytes(raw_bytes: bytes) -> dict[str, Any] | list[Any]:
    decoded = raw_bytes.decode("utf-8-sig", errors="replace")
    payload = json.loads(decoded)
    return sanitize_unicode_payload(payload)
