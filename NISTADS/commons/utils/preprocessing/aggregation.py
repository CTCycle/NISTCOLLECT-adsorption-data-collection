import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
class GuestPropertiesMerge:

    def __init__(self):        
        pass

    #--------------------------------------------------------------------------
    def add_guest_properties(self, adsorption : pd.DataFrame, properties : pd.DataFrame):

        dataset_with_properties = pd.merge(adsorption, properties, 
                                           left_on='adsorbates_name', 
                                           right_on='name', how='inner')
        dataset_with_properties.drop(columns=['name'], inplace=True)
        
        return dataset_with_properties
    
    
# [MERGE DATASETS]
###############################################################################
class AggregateMeasurements:

    def __init__(self):

        self.aggregate_dict = {'temperature' : 'first',                  
                               'adsorbent_name' : 'first',
                               'adsorbates_name' : 'first',                  
                               'molecular_weight' : 'first',
                               'elements': 'first',
                               'heavy_atoms': 'first',
                               'molecular_formula' : 'first',
                               'SMILE': 'first',
                               'H_acceptors' : 'first',
                               'H_donors' : 'first',
                               'pressure_in_Pascal' : list,
                               'uptake_in_mol_g' : list}
        
    #--------------------------------------------------------------------------
    def join_to_string(self, x : list):        
        return ' '.join(x.astype(str))

    #--------------------------------------------------------------------------
    def aggregate_adsorption_measurements(self, dataset : pd.DataFrame):

        grouped_data = dataset.groupby(by='filename').agg(self.aggregate_dict).reset_index()
        grouped_data.drop(columns=['filename'], inplace=True)

        return grouped_data

        