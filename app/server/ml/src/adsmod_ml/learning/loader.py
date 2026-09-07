from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from adsmod_ml.common.constants import PAD_VALUE
from adsmod_ml.common.utils.logger import logger


# [CUSTOM DATASET FOR TORCH DATALOADERS]

###############################################################################
class TorchDictDataset(Dataset):

    # -------------------------------------------------------------------------
    def __init__(
        self,
        inputs: dict[str, np.ndarray],
        outputs: np.ndarray | None,
    ) -> None:
        self.inputs = {key: torch.as_tensor(value) for key, value in inputs.items()}

        self.outputs = torch.as_tensor(outputs) if outputs is not None else None
        self.length = 0
        if self.inputs:
            sample = next(iter(self.inputs.values()))
            self.length = int(sample.shape[0])

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return self.length

    # -------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any] | tuple[dict[str, Any], Any]:
        inputs = {key: value[idx] for key, value in self.inputs.items()}
        if self.outputs is None:
            return inputs
        return inputs, self.outputs[idx]


# [CUSTOM DATA GENERATOR FOR TRAINING]

###############################################################################
class DataLoaderProcessor:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any], metadata: dict) -> None:
        self.normalization_config = metadata.get("normalization_stats", {})
        self.series_length = metadata.get("max_measurements", 30)
        self.smile_length = metadata.get("smile_sequence_size", 30)
        self.smile_vocab = metadata.get("smile_vocabulary", {})
        self.adsorbent_vocab = metadata.get("adsorbent_vocabulary", {})
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def aggregate_inference_data(self, dataset: pd.DataFrame) -> pd.DataFrame:
        aggregate_dict = {
            "temperature": "first",
            "adsorbent_name": "first",
            "adsorbate_name": "first",
            "adsorbate_SMILE": "first",
            "adsorbate_molecular_weight": "first",
            "pressure": lambda x: [float(v) for v in x],
        }

        grouped_data = dataset.groupby(by="filename").agg(aggregate_dict).reset_index()
        grouped_data = grouped_data.drop(columns=["filename"])

        return grouped_data

    # -------------------------------------------------------------------------
    def add_properties_to_inference_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {"adsorbate_SMILE", "adsorbate_molecular_weight"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                "Inference inputs must include Core-provided material properties: "
                + ", ".join(sorted(missing))
            )
        return data

    # -------------------------------------------------------------------------
    def remove_invalid_measurements(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.loc[data["temperature"] >= 0].copy()
        if "pressure" in self.normalization_config:
            max_pressure = self.normalization_config["pressure"]
            data = data.loc[
                (data["pressure"] >= 0) & (data["pressure"] <= max_pressure)
            ]

        return data

    # -------------------------------------------------------------------------
    def normalize_from_references(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.normalization_config:
            return data
        if "temperature" in self.normalization_config:
            data["temperature"] = (
                data["temperature"] / self.normalization_config["temperature"]
            )
        if "adsorbate_molecular_weight" in self.normalization_config:
            data["adsorbate_molecular_weight"] = (
                data["adsorbate_molecular_weight"]
                / self.normalization_config["adsorbate_molecular_weight"]
            )
        if "pressure" in self.normalization_config:
            data["pressure"] = data["pressure"].apply(
                lambda x: [s / self.normalization_config["pressure"] for s in x]
            )

        return data

    # -------------------------------------------------------------------------
    def encode_smile_from_vocabulary(self, smile: str) -> list[Any]:
        encoded_tokens = []
        i = 0
        sorted_tokens = sorted(self.smile_vocab.keys(), key=len, reverse=True)
        while i < len(smile):
            matched = False
            for token in sorted_tokens:
                if smile[i : i + len(token)] == token:
                    encoded_tokens.append(self.smile_vocab[token])
                    i += len(token)
                    matched = True
                    break
            if not matched:
                logger.warning(
                    "SMILE tokenization error: no valid token found in '%s' at position %s",
                    smile,
                    i,
                )
                i += 1

        return encoded_tokens

    # -------------------------------------------------------------------------
    def encode_from_references(self, data: pd.DataFrame) -> pd.DataFrame:
        data["adsorbate_encoded_SMILE"] = data["adsorbate_SMILE"].apply(
            lambda x: self.encode_smile_from_vocabulary(x)
        )
        data["encoded_adsorbent"] = (
            data["adsorbent_name"].str.lower().map(self.adsorbent_vocab)
        )

        return data

    # -------------------------------------------------------------------------
    def apply_padding(self, data: pd.DataFrame) -> pd.DataFrame:
        data["pressure"] = self._pad_sequences(
            data["pressure"].tolist(),
            maxlen=self.series_length,
            value=PAD_VALUE,
            dtype=np.float32,
        )

        data["adsorbate_encoded_SMILE"] = self._pad_sequences(
            data["adsorbate_encoded_SMILE"].tolist(),
            maxlen=self.smile_length,
            value=PAD_VALUE,
            dtype=np.float32,
        )

        return data

    # -------------------------------------------------------------------------
    @staticmethod
    def _pad_sequences(
        sequences: list[Any],
        maxlen: int,
        value: float,
        dtype: Any,
    ) -> list[list[Any]]:
        if not sequences:
            return []
        padded = np.full((len(sequences), maxlen), value, dtype=dtype)
        for idx, sequence in enumerate(sequences):
            if sequence is None:
                continue
            array = np.asarray(sequence, dtype=dtype).reshape(-1)
            if array.size == 0:
                continue
            length = min(array.size, maxlen)
            padded[idx, :length] = array[:length]
        return padded.tolist()

###############################################################################
class SCADSDataLoader:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        configuration: dict[str, Any],
        metadata: dict[str, Any],
        shuffle: bool = True,
    ) -> None:
        self.processor = DataLoaderProcessor(configuration, metadata)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        self.num_workers = configuration.get("dataloader_workers", 0)
        self.prefetch_factor = configuration.get("prefetch_factor", 1)
        self.pin_memory = configuration.get("pin_memory", False)
        self.persistent_workers = configuration.get("persistent_workers", False)
        self.metadata = metadata
        self.configuration = configuration
        self.shuffle = shuffle
        self.output = "adsorbed_amount"

    # -------------------------------------------------------------------------
    def separate_inputs_and_output(
        self, data: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        if data.empty:
            inputs = {
                "state_input": np.empty((0,), dtype=np.float32),
                "chemo_input": np.empty((0,), dtype=np.float32),
                "adsorbent_input": np.empty((0,), dtype=np.int32),
                "adsorbate_input": np.empty(
                    (0, self.processor.smile_length), dtype=np.int32
                ),
                "pressure_input": np.empty(
                    (0, self.processor.series_length), dtype=np.float32
                ),
            }
            return inputs, np.empty((0, self.processor.series_length), dtype=np.float32)

        state = data["temperature"].to_numpy(dtype=np.float32)
        chemo = data["adsorbate_molecular_weight"].to_numpy(dtype=np.float32)
        adsorbent = data["encoded_adsorbent"].to_numpy(dtype=np.int32)
        adsorbate = np.vstack(
            [
                np.asarray(x, dtype=np.int32)
                for x in data["adsorbate_encoded_SMILE"].to_list()
            ]
        )
        pressure = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in data["pressure"].to_list()]
        )
        inputs_dict = {
            "state_input": state,
            "chemo_input": chemo,
            "adsorbent_input": adsorbent,
            "adsorbate_input": adsorbate,
            "pressure_input": pressure,
        }

        output = None
        if self.output in data.columns:
            output = data[self.output]
            output = np.vstack([np.asarray(x, dtype=np.float32) for x in output.values])

        return inputs_dict, output

    # -------------------------------------------------------------------------
    def process_inference_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        processed_data = self.processor.remove_invalid_measurements(data)
        processed_data = self.processor.aggregate_inference_data(processed_data)
        processed_data = self.processor.add_properties_to_inference_inputs(
            processed_data
        )
        processed_data = self.processor.encode_from_references(processed_data)
        processed_data = self.processor.normalize_from_references(processed_data)
        processed_data = self.processor.apply_padding(processed_data)

        return processed_data

    # -------------------------------------------------------------------------
    def build_training_dataloader(
        self, data: pd.DataFrame, batch_size: int | None = None
    ) -> Any:
        batch_size = self.batch_size if batch_size is None else batch_size
        inputs, output = self.separate_inputs_and_output(data)
        dataset = TorchDictDataset(inputs, output)
        loader_settings: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }

        if self.num_workers > 0:
            loader_settings["prefetch_factor"] = self.prefetch_factor
            loader_settings["persistent_workers"] = self.persistent_workers

        return DataLoader(dataset, **loader_settings)

    # -------------------------------------------------------------------------
    def build_inference_dataloader(
        self, data: pd.DataFrame, batch_size: int | None = None
    ) -> Any:
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        processed_data = self.process_inference_inputs(data)
        inputs, _ = self.separate_inputs_and_output(processed_data)
        dataset = TorchDictDataset(inputs, outputs=None)
        loader_settings: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }

        if self.num_workers > 0:
            loader_settings["prefetch_factor"] = self.prefetch_factor
            loader_settings["persistent_workers"] = self.persistent_workers

        return DataLoader(dataset, **loader_settings)

###############################################################################
class SCADSAtomicDataLoader:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        configuration: dict[str, Any],
        metadata: dict[str, Any],
        shuffle: bool = True,
    ) -> None:
        self.processor = DataLoaderProcessor(configuration, metadata)
        self.batch_size = configuration.get("batch_size", 32)
        self.inference_batch_size = configuration.get("inference_batch_size", 32)
        self.num_workers = configuration.get("dataloader_workers", 0)
        self.prefetch_factor = configuration.get("prefetch_factor", 1)
        self.pin_memory = configuration.get("pin_memory", False)
        self.persistent_workers = configuration.get("persistent_workers", False)
        self.metadata = metadata
        self.configuration = configuration
        self.shuffle = shuffle
        self.output = "adsorbed_amount"
        self.smile_length = metadata.get("smile_sequence_size", 20)

    # -------------------------------------------------------------------------
    def expand_to_single_measurements(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            columns = list(data.columns)
            return pd.DataFrame(columns=columns)

        non_sequence_cols = [
            col for col in data.columns if col not in {"pressure", self.output}
        ]
        has_output = self.output in data.columns
        expanded_rows: list[dict[str, Any]] = []

        for _, row in data.iterrows():
            base = {col: row[col] for col in non_sequence_cols}
            pressure_values = row.get("pressure", [])
            if isinstance(pressure_values, np.ndarray):
                pressure_values = pressure_values.tolist()
            elif not isinstance(pressure_values, (list, tuple)):
                pressure_values = [pressure_values]

            uptake_values: list[Any]
            if has_output:
                uptake_values = row.get(self.output, [])
                if isinstance(uptake_values, np.ndarray):
                    uptake_values = uptake_values.tolist()
                elif not isinstance(uptake_values, (list, tuple)):
                    uptake_values = [uptake_values] * len(pressure_values)
            else:
                uptake_values = [None] * len(pressure_values)

            for pressure_value, uptake_value in zip(pressure_values, uptake_values):
                if pd.isna(pressure_value) or pressure_value == PAD_VALUE:
                    continue
                if has_output and uptake_value is not None and pd.isna(uptake_value):
                    continue
                entry = base.copy()
                entry["pressure"] = float(pressure_value)
                if (
                    has_output
                    and uptake_value is not None
                    and uptake_value != PAD_VALUE
                ):
                    entry[self.output] = float(uptake_value)
                expanded_rows.append(entry)

        expanded_df = pd.DataFrame(expanded_rows)
        if expanded_df.empty:
            columns = non_sequence_cols + ["pressure"]
            if has_output:
                columns.append(self.output)
            expanded_df = pd.DataFrame(columns=columns)
            return expanded_df

        required_cols = [
            "temperature",
            "adsorbate_molecular_weight",
            "encoded_adsorbent",
            "pressure",
        ]
        if has_output:
            required_cols.append(self.output)
        available_cols = [col for col in required_cols if col in expanded_df.columns]
        expanded_df = expanded_df.dropna(subset=available_cols).reset_index(drop=True)

        return expanded_df

    # -------------------------------------------------------------------------
    def separate_inputs_and_output(
        self, data: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        expanded = self.expand_to_single_measurements(data)
        if expanded.empty:
            smile_shape = (0, self.smile_length)
            inputs = {
                "adsorbate_input": np.empty(smile_shape, dtype=np.int32),
                "features_input": np.empty((0, 4), dtype=np.float32),
            }
            return inputs, np.empty((0, 1), dtype=np.float32)

        adsorbate = np.vstack(
            [
                np.asarray(x, dtype=np.int32)
                for x in expanded["adsorbate_encoded_SMILE"].to_list()
            ]
        )
        temperature = expanded["temperature"].to_numpy(dtype=np.float32)
        molecular_weight = expanded["adsorbate_molecular_weight"].to_numpy(
            dtype=np.float32
        )
        adsorbent = expanded["encoded_adsorbent"].to_numpy(dtype=np.float32)
        pressure = expanded["pressure"].to_numpy(dtype=np.float32)
        features = np.column_stack((temperature, molecular_weight, adsorbent, pressure))
        inputs_dict = {"adsorbate_input": adsorbate, "features_input": features}

        output = None
        if self.output in expanded.columns:
            output = expanded[self.output].to_numpy(dtype=np.float32).reshape(-1, 1)

        return inputs_dict, output

    # -------------------------------------------------------------------------
    def build_training_dataloader(
        self, data: pd.DataFrame, batch_size: int | None = None
    ) -> Any:
        batch_size = self.batch_size if batch_size is None else batch_size
        inputs, output = self.separate_inputs_and_output(data)
        dataset = TorchDictDataset(inputs, output)
        loader_settings: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }

        if self.num_workers > 0:
            loader_settings["prefetch_factor"] = self.prefetch_factor
            loader_settings["persistent_workers"] = self.persistent_workers

        return DataLoader(dataset, **loader_settings)

    # -------------------------------------------------------------------------
    def process_inference_inputs(self, data: pd.DataFrame) -> pd.DataFrame:
        processed_data = self.processor.remove_invalid_measurements(data)
        processed_data = self.processor.aggregate_inference_data(processed_data)
        processed_data = self.processor.add_properties_to_inference_inputs(
            processed_data
        )
        processed_data = self.processor.encode_from_references(processed_data)
        processed_data = self.processor.normalize_from_references(processed_data)
        processed_data = self.processor.apply_padding(processed_data)
        processed_data = self.expand_to_single_measurements(processed_data)

        return processed_data

    # -------------------------------------------------------------------------
    def build_inference_dataloader(
        self, data: pd.DataFrame, batch_size: int | None = None
    ) -> Any:
        batch_size = self.inference_batch_size if batch_size is None else batch_size
        processed_data = self.process_inference_inputs(data)
        inputs, _ = self.separate_inputs_and_output(processed_data)
        dataset = TorchDictDataset(inputs, outputs=None)
        loader_settings: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": False,
        }

        if self.num_workers > 0:
            loader_settings["prefetch_factor"] = self.prefetch_factor
            loader_settings["persistent_workers"] = self.persistent_workers

        return DataLoader(dataset, **loader_settings)
