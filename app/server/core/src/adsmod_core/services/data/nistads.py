from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import httpx
import pandas as pd

from adsmod_core.common.utils.encoding import (
    decode_json_response_bytes,
    sanitize_dataframe_strings,
)
from adsmod_core.common.utils.logger import logger

###############################################################################
class NISTDatasetBuilder:

    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.raw_drop_cols = [
            "DOI",
            "category",
            "tabular_data",
            "digitizer",
            "isotherm_type",
            "articleSource",
            "concentrationUnits",
            "compositionType",
        ]
        self.single_explode_cols = ["pressure", "adsorbed_amount"]
        self.single_drop_cols = [
            "date",
            "adsorbent",
            "adsorbates",
            "num_guests",
            "isotherm_data",
            "adsorbent_ID",
            "adsorbates_ID",
        ]
        self.binary_explode_cols = [
            "compound_1_pressure",
            "compound_2_pressure",
            "compound_1_adsorption",
            "compound_2_adsorption",
            "compound_1_composition",
            "compound_2_composition",
        ]
        self.binary_drop_cols = [
            "date",
            "adsorbent",
            "adsorbates",
            "num_guests",
            "isotherm_data",
            "adsorbent_ID",
            "adsorbates_ID",
            "adsorbate_name",
            "total_pressure",
            "all_species_data",
            "compound_1_data",
            "compound_2_data",
        ]

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_string_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        return sanitize_dataframe_strings(dataframe)

    # -------------------------------------------------------------------------
    def drop_excluded_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.drop(columns=self.raw_drop_cols, errors="ignore")

    # -------------------------------------------------------------------------
    def split_by_mixture_complexity(
        self, dataframe: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if dataframe.empty:
            return dataframe, dataframe
        dataframe = dataframe.copy()
        dataframe["num_guests"] = dataframe["adsorbates"].str.len()
        single_component = dataframe.loc[dataframe["num_guests"] == 1].copy()
        binary_mixture = dataframe.loc[dataframe["num_guests"] == 2].copy()
        return single_component, binary_mixture

    # -------------------------------------------------------------------------
    def is_single_component(self, dataframe: pd.DataFrame) -> bool:
        return (dataframe["num_guests"] == 1).all()

    # -------------------------------------------------------------------------
    def is_binary_mixture(self, dataframe: pd.DataFrame) -> bool:
        return (dataframe["num_guests"] == 2).all()

    # -------------------------------------------------------------------------
    def add_material_fields(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["adsorbent_ID"] = dataframe["adsorbent"].apply(
            lambda x: (x or {}).get("hashkey")
        )
        dataframe["adsorbent_name"] = dataframe["adsorbent"].apply(
            lambda x: str((x or {}).get("name", "")).lower()
        )
        dataframe["adsorbates_ID"] = dataframe["adsorbates"].apply(
            lambda x: [
                item.get("InChIKey") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["adsorbate_name"] = dataframe["adsorbates"].apply(
            lambda x: [
                str(item.get("name", "")).lower()
                for item in (x or [])
                if isinstance(item, dict)
            ]
        )
        return dataframe

    # -------------------------------------------------------------------------
    def add_single_component_fields(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["pressure"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("pressure") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["adsorbed_amount"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("total_adsorption")
                for item in (x or [])
                if isinstance(item, dict)
            ]
        )
        dataframe["adsorbate_name"] = dataframe["adsorbates"].apply(
            lambda x: (
                str(x[0].get("name", "")).lower() if isinstance(x, list) and x else ""
            )
        )
        return dataframe

    # -------------------------------------------------------------------------
    def add_binary_mixture_fields(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        data_placeholder = {"composition": 1.0, "adsorption": 1.0}
        dataframe["total_pressure"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("pressure") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["all_species_data"] = dataframe["isotherm_data"].apply(
            lambda x: [
                item.get("species_data") for item in (x or []) if isinstance(item, dict)
            ]
        )
        dataframe["compound_1"] = dataframe["adsorbate_name"].apply(
            lambda x: str(x[0]).lower() if isinstance(x, list) and x else ""
        )
        dataframe["compound_2"] = dataframe["adsorbate_name"].apply(
            lambda x: str(x[1]).lower() if isinstance(x, list) and len(x) > 1 else ""
        )
        dataframe["compound_1_data"] = dataframe["all_species_data"].apply(
            lambda x: [item[0] if item else data_placeholder for item in (x or [])]
        )
        dataframe["compound_2_data"] = dataframe["all_species_data"].apply(
            lambda x: [
                item[1] if item and len(item) > 1 else data_placeholder
                for item in (x or [])
            ]
        )
        dataframe["compound_1_composition"] = dataframe["compound_1_data"].apply(
            lambda x: [item.get("composition") for item in (x or [])]
        )
        dataframe["compound_2_composition"] = dataframe["compound_2_data"].apply(
            lambda x: [item.get("composition") for item in (x or [])]
        )
        dataframe["compound_1_pressure"] = dataframe.apply(
            lambda row: [
                a * b
                for a, b in zip(
                    row["compound_1_composition"], row["total_pressure"], strict=False
                )
            ],
            axis=1,
        )
        dataframe["compound_2_pressure"] = dataframe.apply(
            lambda row: [
                a * b
                for a, b in zip(
                    row["compound_2_composition"], row["total_pressure"], strict=False
                )
            ],
            axis=1,
        )
        dataframe["compound_1_adsorption"] = dataframe["compound_1_data"].apply(
            lambda x: [item.get("adsorption") for item in (x or [])]
        )
        dataframe["compound_2_adsorption"] = dataframe["compound_2_data"].apply(
            lambda x: [item.get("adsorption") for item in (x or [])]
        )
        return dataframe

    # -------------------------------------------------------------------------
    def extract_nested_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        dataframe = self.add_material_fields(dataframe)
        if self.is_single_component(dataframe):
            return self.add_single_component_fields(dataframe)
        if self.is_binary_mixture(dataframe):
            return self.add_binary_mixture_fields(dataframe)

        return dataframe

    # -------------------------------------------------------------------------
    def expand_dataset(
        self, single_component: pd.DataFrame, binary_mixture: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if single_component.empty:
            single_dataset = pd.DataFrame()
        else:
            single_dataset = single_component.explode(self.single_explode_cols)
            single_dataset = single_dataset.drop(
                columns=self.single_drop_cols, errors="ignore"
            )
            single_dataset = single_dataset.dropna().reset_index(drop=True)
            single_dataset = single_dataset.rename(
                columns={
                    "filename": "name",
                    "adsorptionUnits": "adsorption_units",
                    "pressureUnits": "pressure_units",
                    "adsorbent_name": "adsorbent",
                    "adsorbate_name": "adsorbate",
                }
            )
            single_dataset = self.normalize_string_columns(single_dataset)

        if binary_mixture.empty:
            binary_dataset = pd.DataFrame()
        else:
            binary_dataset = binary_mixture.explode(self.binary_explode_cols)
            binary_dataset = binary_dataset.drop(
                columns=self.binary_drop_cols, errors="ignore"
            )
            binary_dataset = binary_dataset.dropna().reset_index(drop=True)
            binary_dataset = binary_dataset.rename(
                columns={
                    "filename": "name",
                    "adsorptionUnits": "adsorption_units",
                    "pressureUnits": "pressure_units",
                }
            )
            binary_dataset = self.normalize_string_columns(binary_dataset)

        return single_dataset, binary_dataset

    # -------------------------------------------------------------------------
    def build_datasets(
        self, adsorption_data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if adsorption_data is None or adsorption_data.empty:
            return pd.DataFrame(), pd.DataFrame()

        cleaned = self.drop_excluded_columns(adsorption_data)
        if cleaned.empty:
            return pd.DataFrame(), pd.DataFrame()

        single_component, binary_mixture = self.split_by_mixture_complexity(cleaned)
        if not single_component.empty:
            single_component = self.extract_nested_data(single_component)
        if not binary_mixture.empty:
            binary_mixture = self.extract_nested_data(binary_mixture)

        return self.expand_dataset(single_component, binary_mixture)

###############################################################################
class NISTApiClient:

    # -------------------------------------------------------------------------
    def __init__(self, parallel_tasks: int) -> None:
        self.parallel_tasks = max(1, int(parallel_tasks))
        self.semaphore = asyncio.Semaphore(self.parallel_tasks)
        self.exp_identifier = "filename"
        self.guest_identifier = "InChIKey"
        self.host_identifier = "hashkey"
        self.url_isotherms = "https://adsorption.nist.gov/isodb/api/isotherms.json"
        self.url_guest_index = "https://adsorption.nist.gov/isodb/api/gases.json"
        self.url_host_index = "https://adsorption.nist.gov/matdb/api/materials.json"
        self.extra_guest_columns = [
            "molecular_weight",
            "molecular_formula",
            "smile_code",
        ]
        self.extra_host_columns = [
            "molecular_weight",
            "molecular_formula",
            "smile_code",
        ]

    # -------------------------------------------------------------------------
    async def fetch_json(
        self, client: httpx.AsyncClient, url: str
    ) -> dict[str, Any] | list[Any] | None:
        async with self.semaphore:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Failed to fetch %s: %s", url, exc)
                return None
            try:
                return decode_json_response_bytes(response.content)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("Invalid JSON from %s: %s", url, exc)
                return None

    # -------------------------------------------------------------------------
    async def fetch_multiple(
        self, client: httpx.AsyncClient, urls: list[str]
    ) -> list[Any]:
        if not urls:
            return []
        tasks = [self.fetch_json(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    # -------------------------------------------------------------------------
    async def fetch_experiments_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        payload = await self.fetch_json(client, self.url_isotherms)
        if payload is None:
            raise ValueError("Failed to retrieve NIST adsorption isotherm index.")
        return pd.DataFrame(payload)

    # -------------------------------------------------------------------------
    async def fetch_experiments_data(
        self,
        client: httpx.AsyncClient,
        experiments_index: pd.DataFrame,
        experiments_fraction: float,
    ) -> pd.DataFrame:
        if experiments_index.empty:
            return pd.DataFrame()
        if self.exp_identifier not in experiments_index.columns:
            raise ValueError("NIST isotherm index missing filename column.")
        identifiers = experiments_index[self.exp_identifier].tolist()
        if not identifiers or experiments_fraction <= 0:
            return pd.DataFrame()
        num_samples = int(math.ceil(experiments_fraction * len(identifiers)))
        urls = [
            f"https://adsorption.nist.gov/isodb/api/isotherm/{identifier}.json"
            for identifier in identifiers[:num_samples]
        ]
        results = await self.fetch_multiple(client, urls)
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    async def fetch_experiments_data_by_identifiers(
        self, client: httpx.AsyncClient, identifiers: list[str]
    ) -> pd.DataFrame:
        filtered_identifiers = [identifier for identifier in identifiers if identifier]
        if not filtered_identifiers:
            return pd.DataFrame()
        urls = [
            f"https://adsorption.nist.gov/isodb/api/isotherm/{identifier}.json"
            for identifier in filtered_identifiers
        ]
        results = await self.fetch_multiple(client, urls)
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    async def fetch_materials_index(
        self, client: httpx.AsyncClient
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        guest_payload = await self.fetch_json(client, self.url_guest_index)
        host_payload = await self.fetch_json(client, self.url_host_index)
        guest_data = pd.DataFrame(guest_payload or [])
        host_data = pd.DataFrame(host_payload or [])
        return guest_data, host_data

    # -------------------------------------------------------------------------
    async def fetch_guest_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        payload = await self.fetch_json(client, self.url_guest_index)
        return pd.DataFrame(payload or [])

    # -------------------------------------------------------------------------
    async def fetch_host_index(self, client: httpx.AsyncClient) -> pd.DataFrame:
        payload = await self.fetch_json(client, self.url_host_index)
        return pd.DataFrame(payload or [])

    # -------------------------------------------------------------------------
    def build_material_urls(
        self, identifiers: list[str], fraction: float, url_template: str
    ) -> list[str]:
        if not identifiers or fraction <= 0:
            return []
        num_samples = int(math.ceil(fraction * len(identifiers)))
        return [
            url_template.format(identifier=identifier)
            for identifier in identifiers[:num_samples]
        ]

    # -------------------------------------------------------------------------
    def prepare_materials_frame(
        self,
        results: list[Any],
        identifier_column: str,
        extra_columns: list[str],
        drop_columns: list[str],
    ) -> pd.DataFrame:
        data = pd.DataFrame(results)
        if data.empty:
            return pd.DataFrame(columns=[identifier_column, "name", *extra_columns])
        data = data.drop(columns=drop_columns, errors="ignore")
        data["name"] = data["name"].astype(str).str.lower()
        for col in extra_columns:
            data[col] = pd.NA
        return data

    # -------------------------------------------------------------------------
    async def fetch_material_dataset(
        self,
        client: httpx.AsyncClient,
        index: pd.DataFrame,
        identifier_column: str,
        fraction: float,
        url_template: str,
        extra_columns: list[str],
        drop_columns: list[str],
        label: str,
    ) -> pd.DataFrame:
        if index.empty or fraction <= 0:
            logger.warning("No %s index available for NIST fetch.", label)
            return pd.DataFrame()
        if identifier_column not in index.columns:
            raise ValueError(f"NIST {label} index missing {identifier_column} column.")
        identifiers = index[identifier_column].tolist()
        urls = self.build_material_urls(identifiers, fraction, url_template)
        results = await self.fetch_multiple(client, urls)
        return self.prepare_materials_frame(
            results, identifier_column, extra_columns, drop_columns
        )

    # -------------------------------------------------------------------------
    async def fetch_material_dataset_by_identifiers(
        self,
        client: httpx.AsyncClient,
        identifiers: list[str],
        identifier_column: str,
        url_template: str,
        extra_columns: list[str],
        drop_columns: list[str],
    ) -> pd.DataFrame:
        filtered_identifiers = [identifier for identifier in identifiers if identifier]
        if not filtered_identifiers:
            return pd.DataFrame()
        urls = [
            url_template.format(identifier=identifier)
            for identifier in filtered_identifiers
        ]
        results = await self.fetch_multiple(client, urls)
        return self.prepare_materials_frame(
            results,
            identifier_column=identifier_column,
            extra_columns=extra_columns,
            drop_columns=drop_columns,
        )

    # -------------------------------------------------------------------------
    async def fetch_materials_data(
        self,
        client: httpx.AsyncClient,
        guest_index: pd.DataFrame,
        host_index: pd.DataFrame,
        guest_fraction: float,
        host_fraction: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        guest_data = await self.fetch_material_dataset(
            client=client,
            index=guest_index,
            identifier_column=self.guest_identifier,
            fraction=guest_fraction,
            url_template="https://adsorption.nist.gov/isodb/api/gas/{identifier}.json",
            extra_columns=self.extra_guest_columns,
            drop_columns=["synonyms"],
            label="guest",
        )
        host_data = await self.fetch_material_dataset(
            client=client,
            index=host_index,
            identifier_column=self.host_identifier,
            fraction=host_fraction,
            url_template="https://adsorption.nist.gov/matdb/api/material/{identifier}.json",
            extra_columns=self.extra_host_columns,
            drop_columns=["External_Resources", "synonyms"],
            label="host",
        )

        return guest_data, host_data


__all__ = [
    "NISTDatasetBuilder",
    "NISTApiClient",
]
