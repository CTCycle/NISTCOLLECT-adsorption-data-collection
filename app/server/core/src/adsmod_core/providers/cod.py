from __future__ import annotations

import re
import shlex
from typing import Any

from adsmod_core.providers.public_data import (
    ProviderCapability,
    ProviderNotFoundError,
    ProviderUnavailableError,
    RetryingHttpProvider,
)


_NUMBER_PREFIX = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")
_ELEMENT_PREFIX = re.compile(r"^[A-Z][a-z]?")


###############################################################################
class CODProvider(RetryingHttpProvider):
    key = "cod"
    name = "Crystallography Open Database"
    description = (
        "Open crystallographic structures with unit-cell metadata, atomic coordinates, "
        "CIF files, and publication references."
    )
    homepage_url = "https://www.crystallography.net/cod/"
    license_name = "CC0 1.0"
    license_url = "https://creativecommons.org/publicdomain/zero/1.0/"
    terms_url = "https://wiki.crystallography.net/howtoobtaincod/"
    capabilities = (
        ProviderCapability.MATERIALS,
        ProviderCapability.STRUCTURES,
        ProviderCapability.REFERENCES,
    )
    search_url = "https://www.crystallography.net/cod/result"
    entry_base_url = "https://www.crystallography.net/cod"

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        request_timeout_seconds: float,
        retry_attempts: int,
        max_interactive_results: int,
    ) -> None:
        super().__init__(
            parallel_requests=1,
            request_timeout_seconds=request_timeout_seconds,
            retry_attempts=retry_attempts,
        )
        self.max_interactive_results = max(1, int(max_interactive_results))

    # -------------------------------------------------------------------------
    async def _health_request(self) -> None:
        await self._request(
            "GET",
            self.search_url,
            params={"id": "1000000", "format": "count"},
            headers={"Accept": "text/plain"},
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _float(value: Any) -> float | None:
        if value in (None, "", "?", "."):
            return None
        match = _NUMBER_PREFIX.match(str(value).strip())
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _integer(value: Any) -> int | None:
        number = CODProvider._float(value)
        return int(number) if number is not None else None

    # -------------------------------------------------------------------------
    @staticmethod
    def _text(row: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = row.get(key)
            if value not in (None, "", "?", "."):
                return str(value).strip()
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _has_coordinates(row: dict[str, Any]) -> bool:
        flags = str(row.get("flags") or "").lower()
        return "coordinates" in flags or str(row.get("has_coordinates") or "").lower() in {
            "1",
            "true",
            "yes",
        }

    # -------------------------------------------------------------------------
    def normalize_result(self, row: dict[str, Any]) -> dict[str, Any]:
        cod_id = str(row.get("file") or row.get("id") or "").strip()
        if not cod_id:
            raise ProviderUnavailableError("COD returned a structure without a COD ID.")
        return {
            "cod_id": cod_id,
            "name": self._text(row, "chemname", "mineral", "name", "text"),
            "formula": self._text(row, "formula", "calcformula"),
            "space_group": self._text(row, "sg", "spacegroup"),
            "space_group_number": self._integer(row.get("sgNumber") or row.get("sg_number")),
            "cell_a_angstrom": self._float(row.get("a")),
            "cell_b_angstrom": self._float(row.get("b")),
            "cell_c_angstrom": self._float(row.get("c")),
            "cell_alpha_deg": self._float(row.get("alpha")),
            "cell_beta_deg": self._float(row.get("beta")),
            "cell_gamma_deg": self._float(row.get("gamma")),
            "cell_volume_angstrom3": self._float(row.get("vol")),
            "doi": self._text(row, "doi"),
            "year": self._integer(row.get("year")),
            "journal": self._text(row, "journal"),
            "title": self._text(row, "title"),
            "has_coordinates": self._has_coordinates(row),
            "source_version": self._text(row, "svnrevision", "revision"),
            "source_url": f"{self.entry_base_url}/{cod_id}.html",
            "cif_url": f"{self.entry_base_url}/{cod_id}.cif",
            "raw_metadata": row,
        }

    # -------------------------------------------------------------------------
    async def search(
        self,
        *,
        text: str | None = None,
        formula: str | None = None,
        cod_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if cod_id:
            params["id"] = cod_id.strip()
        if formula:
            params["formula"] = formula.strip()
        if text:
            normalized_text = text.strip()
            if len(normalized_text) < 3:
                raise ValueError("COD text search requires at least three characters.")
            params["text"] = normalized_text
        if not params:
            raise ValueError("Provide a COD ID, formula, or text query.")

        count_response = await self._request(
            "GET",
            self.search_url,
            params={**params, "format": "count"},
            headers={"Accept": "text/plain"},
        )
        try:
            count = int(count_response.text.strip())
        except ValueError as exc:
            raise ProviderUnavailableError("COD returned an invalid result count.") from exc
        if count == 0:
            return []
        if count > self.max_interactive_results:
            raise ValueError(
                f"COD query matches {count} records. Narrow the formula or text query "
                f"to {self.max_interactive_results} records or fewer before retrieval."
            )

        response = await self._request(
            "GET",
            self.search_url,
            params={**params, "format": "json"},
        )
        payload = response.json()
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("results") or payload.get("records") or []
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ProviderUnavailableError("COD returned an unexpected JSON result shape.")
        return [self.normalize_result(dict(row)) for row in rows if isinstance(row, dict)]

    # -------------------------------------------------------------------------
    async def fetch_record(self, cod_id: str) -> tuple[dict[str, Any], str]:
        matches = await self.search(cod_id=cod_id)
        if not matches:
            raise ProviderNotFoundError(f"COD entry {cod_id} was not found.")
        metadata = matches[0]
        cif_response = await self._request(
            "GET",
            metadata["cif_url"],
            headers={"Accept": "chemical/x-cif, text/plain;q=0.9, */*;q=0.5"},
        )
        cif_text = cif_response.text
        if not cif_text.strip():
            raise ProviderUnavailableError(f"COD entry {cod_id} returned an empty CIF file.")
        return metadata, cif_text

    # -------------------------------------------------------------------------
    @classmethod
    def parse_atoms(cls, cif_text: str) -> list[dict[str, Any]]:
        """Parse the standard atom-site fractional-coordinate loop from a CIF.

        ADSMOD intentionally stores only stable atom-site fields in normalized tables.
        The original CIF is retained verbatim for provenance and specialist tooling.
        """

        lines = cif_text.splitlines()
        atoms: list[dict[str, Any]] = []
        index = 0
        while index < len(lines):
            if lines[index].strip().lower() != "loop_":
                index += 1
                continue
            index += 1
            headers: list[str] = []
            while index < len(lines) and lines[index].lstrip().startswith("_"):
                headers.append(lines[index].strip().split()[0])
                index += 1
            lowered = [header.lower() for header in headers]
            required = (
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
            )
            if not all(name in lowered for name in required):
                while index < len(lines):
                    current = lines[index].strip()
                    if (
                        current.lower() == "loop_"
                        or current.startswith("_")
                        or current.lower().startswith("data_")
                    ):
                        break
                    index += 1
                continue

            positions = {name: lowered.index(name) for name in required}
            label_position = (
                lowered.index("_atom_site_label")
                if "_atom_site_label" in lowered
                else None
            )
            type_position = (
                lowered.index("_atom_site_type_symbol")
                if "_atom_site_type_symbol" in lowered
                else None
            )
            occupancy_position = (
                lowered.index("_atom_site_occupancy")
                if "_atom_site_occupancy" in lowered
                else None
            )
            while index < len(lines):
                current = lines[index].strip()
                if not current or current.startswith("#"):
                    index += 1
                    continue
                if (
                    current.lower() == "loop_"
                    or current.startswith("_")
                    or current.lower().startswith("data_")
                ):
                    break
                try:
                    values = shlex.split(current, comments=False, posix=True)
                except ValueError:
                    index += 1
                    continue
                if len(values) < len(headers):
                    index += 1
                    continue
                x = cls._float(values[positions["_atom_site_fract_x"]])
                y = cls._float(values[positions["_atom_site_fract_y"]])
                z = cls._float(values[positions["_atom_site_fract_z"]])
                if x is None or y is None or z is None:
                    index += 1
                    continue
                label = (
                    values[label_position]
                    if label_position is not None
                    else f"A{len(atoms) + 1}"
                )
                element = values[type_position] if type_position is not None else label
                element_match = _ELEMENT_PREFIX.match(str(element)) or _ELEMENT_PREFIX.match(
                    str(label)
                )
                element_symbol = (
                    element_match.group(0) if element_match else str(element)[:8]
                )
                occupancy = (
                    cls._float(values[occupancy_position])
                    if occupancy_position is not None
                    else None
                )
                atoms.append(
                    {
                        "sequence_index": len(atoms),
                        "label": str(label),
                        "element": element_symbol,
                        "fractional_x": x,
                        "fractional_y": y,
                        "fractional_z": z,
                        "occupancy": occupancy,
                    }
                )
                index += 1
            if atoms:
                return atoms
        return atoms


__all__ = ["CODProvider"]
