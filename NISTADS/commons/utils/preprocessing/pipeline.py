import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from NISTADS.commons.utils.dataloader.serializer import get_datasets
from NISTADS.commons.utils.preprocessing.filtering import filter_outside_boundaries 
from NISTADS.commons.utils.preprocessing.splitting import DatasetSplit 
from NISTADS.commons.utils.preprocessing.aggregation import GuestPropertiesMerge, AggregateMeasurements
from NISTADS.commons.utils.preprocessing.conversion import PressureConversion, UptakeConversion
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
class ProcessPipeline:

    def __init__(self):

        self.SCADS, self.properties = get_datasets()

        self.sample_size = CONFIG["dataset"]["SAMPLE_SIZE"]
        self.max_pressure = CONFIG["dataset"]["MAX_PRESSURE"] 
        self.max_uptake = CONFIG["dataset"]["MAX_UPTAKE"] 

        self.merger = GuestPropertiesMerge()
        self.aggregator = AggregateMeasurements()
        self.splitter = DatasetSplit()
        self.P_converter = PressureConversion()
        self.Q_converter = UptakeConversion()

    #--------------------------------------------------------------------------
    def preprocess_and_split_dataset(self):

        filter_dataset = filter_outside_boundaries(self.SCADS, self.max_pressure, self.max_uptake)
        processed_data = self.merger.merge_guest_properties(filter_dataset, self.properties)        
        processed_data = self.P_converter.convert_data(processed_data)
        processed_data = self.Q_converter.convert_data(processed_data)
        aggregated_data = self.aggregator.aggregate_experiment_measurements(processed_data)
        train, validation = self.splitter.split_dataset(aggregated_data)

        return train, validation
        