import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
class GuestPropertiesMerge:

    def __init__(self, properties : pd.DataFrame, adsorption : pd.DataFrame):
        
        self.properties = properties
        self.adsorption = adsorption

    #--------------------------------------------------------------------------
    def merge_guest_properties(self):

        dataset_with_properties = pd.merge(self.adsorption, self.properties, 
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
                                'complexity' : 'first',                  
                                'mol_weight' : 'first',
                                'covalent_units' : 'first',
                                'H_acceptors' : 'first',
                                'H_donors' : 'first',
                                'heavy_atoms' : 'first', 
                                'pressure_in_Pascal' : join_str,
                                'uptake_in_mol_g' : join_str}

    #--------------------------------------------------------------------------
    def merge_guest_properties(self):

        dataset_with_properties = pd.merge(self.SCADS, self.properties, 
                                           left_on='adsorbates_name', 
                                           right_on='name', how='inner')
        dataset_with_properties.drop(columns=['name'], inplace=True)
        
        return dataset_with_properties


