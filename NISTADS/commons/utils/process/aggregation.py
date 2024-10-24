import os
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
class GuestPropertiesMerge:

    def __init__(self):        
        self.selected_properties = ['name', 'heavy_atoms', 'elements', 'molecular_weight',
                                    'molecular_formula', 'SMILE', 'H_acceptors', 'H_donors']

    #--------------------------------------------------------------------------
    def add_guest_properties(self, adsorption : pd.DataFrame, properties : pd.DataFrame):

        dataset_with_properties = pd.merge(adsorption, properties[self.selected_properties], 
                                           left_on='adsorbate_name', 
                                           right_on='name', how='inner')
        dataset_with_properties.drop(columns=['name'], inplace=True)
        
        return dataset_with_properties
    
    
# [MERGE DATASETS]
###############################################################################
class AggregateMeasurements:

    def __init__(self):

        self.aggregate_dict = {'temperature' : 'first',                  
                               'adsorbent_name' : 'first',
                               'adsorbate_name' : 'first',
                               'pressureUnits' : 'first',
                               'adsorptionUnits' : 'first',                            
                               'pressure' : list,
                               'adsorbed_amount' : list}   

    #--------------------------------------------------------------------------
    def aggregate_adsorption_measurements(self, dataset : pd.DataFrame):

        grouped_data = dataset.groupby(by='filename').agg(self.aggregate_dict).reset_index()
        grouped_data.drop(columns=['filename'], inplace=True)

        return grouped_data

        