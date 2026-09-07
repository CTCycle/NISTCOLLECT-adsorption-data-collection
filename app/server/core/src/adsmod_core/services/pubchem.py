"""NIST enrichment adapter backed by the canonical PubChem provider."""

from __future__ import annotations

import asyncio
from typing import Any

from adsmod_core.providers.public_data import ProviderError
from adsmod_core.providers.pubchem import PubChemProvider

###############################################################################
class PubChemClient:
    """Adapt normalized PubChem records to the NIST enrichment row contract."""

    # -------------------------------------------------------------------------
    def __init__(self, parallel_tasks: int) -> None:
        self.provider = PubChemProvider(parallel_requests=parallel_tasks)

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_name(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    # -------------------------------------------------------------------------
    @staticmethod
    def is_not_found_error(message: str) -> bool:
        lowered = message.lower()
        return "not found" in lowered or "404" in lowered

    # -------------------------------------------------------------------------
    @staticmethod
    def is_retryable_error(message: str) -> bool:
        lowered = message.lower()
        return any(
            token in lowered
            for token in ("rate limit", "too many requests", "503", "timeout", "unavailable")
        )

    # -------------------------------------------------------------------------
    async def fetch_properties_for_name(self, name: str) -> dict[str, Any]:
        normalized_name = self.normalize_name(name)
        if not normalized_name:
            return self._empty(normalized_name)
        try:
            payload = await self.provider.resolve(normalized_name)
        except (ProviderError, ValueError):
            return self._empty(normalized_name)
        return {
            "name": normalized_name,
            "molecular_weight": payload.get("molecular_weight"),
            "molecular_formula": payload.get("formula"),
            "smile": payload.get("smiles"),
        }

    # -------------------------------------------------------------------------
    async def fetch_properties_for_names(
        self, names: list[str]
    ) -> list[dict[str, Any]]:
        if not names:
            return []
        return await asyncio.gather(
            *(self.fetch_properties_for_name(name) for name in names)
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _empty(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "molecular_weight": None,
            "molecular_formula": None,
            "smile": None,
        }


__all__ = ["PubChemClient"]
