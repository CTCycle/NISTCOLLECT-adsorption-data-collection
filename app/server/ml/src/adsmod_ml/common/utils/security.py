from __future__ import annotations

import re
from os import PathLike
from pathlib import Path


SQL_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
CHECKPOINT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

###############################################################################
def ensure_safe_sql_identifier(identifier: str, field_name: str = "identifier") -> str:
    normalized = str(identifier).strip()
    if not normalized or not SQL_IDENTIFIER_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid {field_name}.")
    return normalized

###############################################################################
def ensure_safe_checkpoint_name(checkpoint_name: str) -> str:
    normalized = str(checkpoint_name).strip()
    if not normalized or not CHECKPOINT_NAME_PATTERN.fullmatch(normalized):
        raise ValueError("Invalid checkpoint name.")
    return normalized

###############################################################################
def resolve_checkpoint_path(
    base_path: str | PathLike[str], checkpoint_name: str
) -> str:
    safe_name = ensure_safe_checkpoint_name(checkpoint_name)
    absolute_base = Path(base_path).resolve()
    resolved = (absolute_base / safe_name).resolve()
    try:
        resolved.relative_to(absolute_base)
    except ValueError:
        raise ValueError("Invalid checkpoint path.")
    return str(resolved)
