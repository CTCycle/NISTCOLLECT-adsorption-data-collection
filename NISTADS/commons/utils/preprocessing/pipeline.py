import os
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from NISTADS.commons.utils.dataloader.serializer import get_datasets
from NISTADS.commons.utils.preprocessing.filtering import filter_outside_boundaries 
from NISTADS.commons.utils.preprocessing.splitting import DatasetSplit 
from NISTADS.commons.utils.preprocessing.aggregation import GuestPropertiesMerge, AggregateMeasurements
from NISTADS.commons.utils.preprocessing.conversion import PressureConversion, UptakeConversion
from NISTADS.commons.utils.preprocessing.sequences import SequenceProcessing
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
class ProcessPipeline:

    def __init__(self):

        logger.info(f'Loading SCADS dataset from {DATA_PATH}')
        self.adsorption, self.guests, self.hosts = get_datasets()        

        self.sample_size = CONFIG["dataset"]["SAMPLE_SIZE"]
        self.max_pressure = CONFIG["dataset"]["MAX_PRESSURE"] 
        self.max_uptake = CONFIG["dataset"]["MAX_UPTAKE"]         

        self.merger = GuestPropertiesMerge()
        self.aggregator = AggregateMeasurements()
        self.splitter = DatasetSplit()
        self.sequencer = SequenceProcessing()
        self.P_converter = PressureConversion()
        self.Q_converter = UptakeConversion()

    #--------------------------------------------------------------------------
    def adsorption_dataset_pipeline(self):


        processed_data = filter_outside_boundaries(processed_data, self.max_pressure, self.max_uptake) 

        # processed_data = self.merger.add_guest_properties(self.adsorption, self.guests)
        # processed_data = self.P_converter.convert_data(processed_data)
        # processed_data = self.Q_converter.convert_data(processed_data)
        # processed_data = filter_outside_boundaries(processed_data, self.max_pressure, self.max_uptake) 
        # aggregated_data = self.aggregator.aggregate_adsorption_measurements(processed_data)        
        # train_X, val_X, train_Y, val_Y = self.splitter.split_train_and_validation(aggregated_data)

        # train_exp, train_guest, train_host, train_pressure = self.splitter.isolate_inputs(train_X)
        # val_exp, val_guest, val_host, val_pressure = self.splitter.isolate_inputs(val_X)

        # processed_data = {'train inputs' : (train_exp, train_guest, train_host, train_pressure),
        #                   'train output' : train_Y,
        #                   'validation' : (val_exp, val_guest, val_host, val_pressure),
        #                   'validation output' :val_Y}

        return processed_data
    
    #--------------------------------------------------------------------------
    def materials_dataset_pipeline(self):

        pass