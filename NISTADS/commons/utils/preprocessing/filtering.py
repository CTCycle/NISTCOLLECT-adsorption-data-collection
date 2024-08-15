import pandas as pd
from tqdm import tqdm
tqdm.pandas()      

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger

# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values 
###############################################################################
class DataFilter:

    def __init__(self):

        self.P_TARGET_COL = 'pressure_in_Pascal'
        
    #--------------------------------------------------------------------------
    def filter_outside_boundaries(self, dataset : pd.DataFrame, max_pressure, max_uptake):

        dataset = dataset[dataset['temperature'].astype(int) > 0]
        dataset = dataset[dataset['pressure'].astype(float).between(0.0, max_pressure)]
        dataset = dataset[dataset['adsorbed_amount'].astype(float).between(0.0, max_uptake)] 

        return dataset
    
    #--------------------------------------------------------------------------
    def filter_by_measurements_count(self, dataset : pd.DataFrame, max_points, min_points):
        
        dataset = dataset[dataset[self.P_TARGET_COL].apply(lambda x: min_points <= len(x) <= max_points)]

        return dataset



    


    

        
    
    

    
 

    