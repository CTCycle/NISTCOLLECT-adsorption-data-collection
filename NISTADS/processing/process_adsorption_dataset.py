# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [IMPORT LIBRARIES]
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import get_datasets
from NISTADS.commons.utils.datamaker.properties import FetchMolecularProperties
from NISTADS.commons.utils.preprocessing.filtering import DataFilter
from NISTADS.commons.utils.preprocessing.splitting import DatasetSplit 
from NISTADS.commons.utils.preprocessing.aggregation import GuestPropertiesMerge, AggregateMeasurements
from NISTADS.commons.utils.preprocessing.conversion import PressureConversion, UptakeConversion
from NISTADS.commons.utils.preprocessing.sequences import SequenceProcessing
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, retrieve and merge molecular properties 
    logger.info(f'Loading SCADS dataset from {DATA_PATH}')
    adsorption, guests, hosts = get_datasets() 

    
    

    # 2. [LOAD AND ENRICH DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, retrieve and merge molecular properties 
    filter = DataFilter()
    logger.info(f'{adsorption.shape[0]} measurements detected in the dataset')
    filtered_data = filter.filter_outside_boundaries(self.adsorption, self.max_pressure, self.max_uptake)
    measurements_out_of_bound = self.adsorption.shape[0] - filtered_data.shape[0] 

    processed_data = self.aggregator.aggregate_adsorption_measurements(filtered_data)
    processed_data = self.merger.add_guest_properties(processed_data, self.guests)
    
    processed_data = self.P_converter.convert_data(processed_data)
    processed_data = self.Q_converter.convert_data(processed_data)
    processed_data = self.sequencer.remove_leading_zeros(processed_data)
    processed_data = self.filter.filter_by_measurements_count(processed_data, self.max_points, self.min_points)
    num_of_experiments = processed_data.shape[0]
    
    # train_X, val_X, train_Y, val_Y = self.splitter.split_train_and_validation(aggregated_data)

    # train_exp, train_guest, train_host, train_pressure = self.splitter.isolate_inputs(train_X)
    # val_exp, val_guest, val_host, val_pressure = self.splitter.isolate_inputs(val_X)

    # processed_data = {'train inputs' : (train_exp, train_guest, train_host, train_pressure),
    #                   'train output' : train_Y,
    #                   'validation' : (val_exp, val_guest, val_host, val_pressure),
    #                   'validation output' :val_Y}












