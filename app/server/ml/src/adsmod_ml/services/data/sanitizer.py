from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

from adsmod_ml.common.utils.logger import logger

###############################################################################
class AggregateDatasets:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.guest_properties = [
            "name",
            "molecular_weight",
            "smile_code",
        ]
        self.host_properties = ["name"]
        self.configuration = configuration

    # -------------------------------------------------------------------------
    def join_materials_properties(
        self, adsorption: pd.DataFrame, guests: pd.DataFrame, hosts: pd.DataFrame
    ) -> pd.DataFrame:
        merged_data = adsorption.merge(
            guests[self.guest_properties],
            left_on="adsorbate_name",
            right_on="name",
            how="inner",
            suffixes=("", "guest"),
        ).drop(columns=["name"])
        merged_data = merged_data.rename(
            columns={
                "molecular_weight": "adsorbate_molecular_weight",
                "smile_code": "adsorbate_SMILE",
            }
        )

        merged_data = merged_data.merge(
            hosts[self.host_properties],
            left_on="adsorbent_name",
            right_on="name",
            how="left",
            suffixes=("", "host"),
        ).drop(columns=["name"])

        return merged_data

    # -------------------------------------------------------------------------
    def aggregate_adsorption_measurements(self, dataset: pd.DataFrame) -> pd.DataFrame:
        aggregate_dict = {
            "temperature": "first",
            "adsorbent_name": "first",
            "adsorbate_name": "first",
            "pressure_units": "first",
            "adsorption_units": "first",
            "pressure": lambda x: [float(v) for v in x],
            "adsorbed_amount": lambda x: [float(v) for v in x],
        }

        grouped_data = dataset.groupby(by="filename").agg(aggregate_dict).reset_index()

        return grouped_data

###############################################################################
class DataSanitizer:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.separator = " "
        self.P_TARGET_COL = "pressure"
        self.Q_TARGET_COL = "adsorbed_amount"
        self.T_TARGET_COL = "temperature"
        self.adsorbate_col = "adsorbate_name"
        self.adsorbent_col = "adsorbent_name"
        self.max_pressure = configuration.get("max_pressure", 10000) * 1000
        self.max_uptake = configuration.get("max_uptake", 20)
        self.configuration = configuration
        self.included_cols = [
            "filename",
            "temperature",
            "pressure",
            "adsorbed_amount",
            "encoded_adsorbent",
            "adsorbate_molecular_weight",
            "adsorbate_encoded_SMILE",
            "split",
        ]

    # -------------------------------------------------------------------------
    def is_convertible_to_float(self, value: Any) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    # -------------------------------------------------------------------------
    def filter_elements_outside_boundaries(
        self, row: pd.Series[list[Any]]
    ) -> pd.Series[list[Any]]:
        p_list = row[self.P_TARGET_COL]
        q_list = row[self.Q_TARGET_COL]
        filtered_p = []
        filtered_q = []
        final_p = []
        final_q = []
        for p, q in zip(p_list, q_list):
            if 0.0 <= p <= self.max_pressure:
                filtered_p.append(p)
                filtered_q.append(q)

        for p, q in zip(filtered_p, filtered_q):
            if 0.0 <= q <= self.max_uptake:
                final_p.append(p)
                final_q.append(q)

        return pd.Series({self.P_TARGET_COL: final_p, self.Q_TARGET_COL: final_q})

    # -------------------------------------------------------------------------
    def exclude_OOB_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.loc[dataset[self.T_TARGET_COL].astype(int) > 0].copy()
        filtered_series = dataset.apply(self.filter_elements_outside_boundaries, axis=1)
        dataset[self.P_TARGET_COL] = filtered_series[self.P_TARGET_COL]
        dataset[self.Q_TARGET_COL] = filtered_series[self.Q_TARGET_COL]

        return dataset

    # -------------------------------------------------------------------------
    def isolate_processed_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.loc[:, self.included_cols].copy()
        dataset = dataset.dropna().reset_index(drop=True)
        return dataset

###############################################################################
class AdsorbentEncoder:

    # -------------------------------------------------------------------------
    def __init__(
        self, configuration: dict[str, Any], train_dataset: pd.DataFrame
    ) -> None:
        self.unknown_class_index = -1
        self.norm_columns = "adsorbent_name"
        self.configuration = configuration

        self.scaler = LabelEncoder()
        self.scaler.fit(train_dataset[self.norm_columns])
        self.mapping = {label: idx for idx, label in enumerate(self.scaler.classes_)}

    # -------------------------------------------------------------------------
    def encode_adsorbents_by_name(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["encoded_adsorbent"] = (
            dataset[self.norm_columns]
            .map(self.mapping)
            .fillna(self.unknown_class_index)
            .astype(int)
        )

        return dataset

    # -------------------------------------------------------------------------
    def encode_adsorbents_from_vocabulary(
        self, dataset: pd.DataFrame, vocabulary: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[Any, str]]:
        mapping = {label: idx for idx, label in vocabulary.items()}
        dataset["encoded_adsorbent"] = (
            dataset[self.norm_columns]
            .map(vocabulary)
            .fillna(self.unknown_class_index)
            .astype(int)
        )

        return dataset, mapping

###############################################################################
class FeatureNormalizer:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        configuration: dict[str, Any],
        train_dataset: pd.DataFrame,
        statistics: dict | None = None,
    ) -> None:
        self.P_COL = "pressure"
        self.Q_COL = "adsorbed_amount"
        self.norm_columns = ["temperature", "adsorbate_molecular_weight"]
        self.configuration = configuration
        self.statistics = (
            self.get_normalization_parameters(train_dataset)
            if statistics is None and train_dataset is not None
            else statistics
        )

    # -------------------------------------------------------------------------
    def get_normalization_parameters(self, train_data: pd.DataFrame) -> dict[str, Any]:
        statistics = {}
        for col in self.norm_columns:
            statistics[col] = train_data[col].astype(float).max()

        p_values = np.concatenate(train_data[self.P_COL].to_numpy())
        q_values = np.concatenate(train_data[self.Q_COL].to_numpy())
        statistics[self.P_COL] = p_values.max()
        statistics[self.Q_COL] = q_values.max()

        return statistics

    # -------------------------------------------------------------------------
    def normalize_molecular_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.statistics is not None:
            norm_cols_stats = {
                k: v for k, v in self.statistics.items() if k in self.norm_columns
            }
            for k, v in norm_cols_stats.items():
                dataset[k] = dataset[k].astype(float) / v

        return dataset

    # -------------------------------------------------------------------------
    def PQ_series_normalization(self, dataset: pd.DataFrame) -> pd.DataFrame:
        if self.statistics is not None:
            P_max = self.statistics[self.P_COL]
            Q_max = self.statistics[self.Q_COL]
            dataset[self.P_COL] = dataset[self.P_COL].apply(
                lambda x: [(v / P_max) for v in x]
            )
            dataset[self.Q_COL] = dataset[self.Q_COL].apply(
                lambda x: [(v / Q_max) for v in x]
            )

        return dataset

###############################################################################
class TrainValidationSplit:

    # -------------------------------------------------------------------------
    def __init__(self, configuration: dict[str, Any]) -> None:
        self.P_COL = "pressure"
        self.Q_COL = "adsorbed_amount"
        self.adsorbate_col = "adsorbate_name"
        self.adsorbent_col = "adsorbent_name"

        self.validation_size = configuration.get("validation_size", 0.2)
        self.seed = configuration.get("split_seed", 42)
        self.train_size = 1.0 - self.validation_size

    # -------------------------------------------------------------------------
    def remove_underpopulated_classes(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["combination"] = (
            dataset[self.adsorbate_col].astype(str)
            + "_"
            + dataset[self.adsorbent_col].astype(str)
        )
        combo_counts = dataset["combination"].value_counts()
        valid_combinations = combo_counts[combo_counts >= 2].index
        dataset = dataset.loc[dataset["combination"].isin(valid_combinations)].copy()

        return dataset

    # -------------------------------------------------------------------------
    def split_train_and_validation(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = self.remove_underpopulated_classes(dataset).reset_index(drop=True)
        combination_classes = dataset["combination"]
        n_samples = len(dataset)

        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.validation_size, random_state=self.seed
            )
            train_idx, val_idx = next(splitter.split(dataset, combination_classes))
        except Exception:
            logger.warning(
                "Validation set too small for the number of classes. Falling back to default split"
            )
            train_idx, val_idx = train_test_split(
                range(n_samples),
                test_size=self.validation_size,
                random_state=self.seed,
                shuffle=True,
            )

        dataset = dataset.drop(columns=["combination"])
        dataset.loc[train_idx, "split"] = "train"
        dataset.loc[val_idx, "split"] = "validation"

        return dataset
