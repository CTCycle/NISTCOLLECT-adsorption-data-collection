from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, DateTime, TypeDecorator

###############################################################################
def normalize_identity(value: str) -> str:
    """Return the application-owned identity representation used by unique keys."""
    normalized = " ".join(value.strip().casefold().split())
    if not normalized:
        raise ValueError("Identity values must not be empty.")
    return normalized

###############################################################################
class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                return datetime(1970, 1, 1)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("UTCDateTime values must be timezone-aware.")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    # -------------------------------------------------------------------------
    def process_result_value(
        self, value: datetime | None, dialect: Any
    ) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=timezone.utc)

###############################################################################
class _StrictJSON(TypeDecorator[Any]):
    impl = JSON
    cache_ok = True

    # -------------------------------------------------------------------------
    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB)
        return dialect.type_descriptor(JSON)

###############################################################################
class JSONList(_StrictJSON):
    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(self, value: Any, dialect: Any) -> list[Any] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError("JSONList values must be lists.")
        return value

    # -------------------------------------------------------------------------
    def process_result_value(self, value: Any, dialect: Any) -> list[Any] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError("JSONList values must be lists.")
        return value

###############################################################################
class JSONMapping(_StrictJSON):
    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(self, value: Any, dialect: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("JSONMapping values must be mappings.")
        return dict(value)

    # -------------------------------------------------------------------------
    def process_result_value(self, value: Any, dialect: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("JSONMapping values must be mappings.")
        return value

###############################################################################
class JSONSequence(_StrictJSON):
    """JSON storage that rejects non-list payloads when values are read back."""

    cache_ok = True

    # -------------------------------------------------------------------------
    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None or isinstance(value, (list, str)):
            return value
        raise TypeError("JSONSequence values must be lists or JSON strings.")

    # -------------------------------------------------------------------------
    def process_result_value(self, value: Any, dialect: Any) -> list[Any] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("Invalid JSONSequence payload")
        return value
