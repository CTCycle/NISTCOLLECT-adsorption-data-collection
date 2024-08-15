import pandas as pd
from tqdm import tqdm
tqdm.pandas()      


# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values 
###############################################################################
def filter_outside_boundaries(dataset : pd.DataFrame, max_pressure, max_uptake):

    dataset = dataset[dataset['temperature'].astype(int) > 0]
    dataset = dataset[dataset['pressure'].astype(float).between(0.0, max_pressure)]
    dataset = dataset[dataset['adsorbed_amount'].astype(float).between(0.0, max_uptake)] 

    return dataset


# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values 
###############################################################################
def drop_not_significant_columns(dataset : pd.DataFrame):

    drop_cols = ['filename', 'adsorptionUnits', 'pressureUnits',        
                 'compositionType', 'numGuests', 'composition', 'atoms', 'bonds']
    
    dataset.drop(columns=drop_cols, inplace=True)

    return dataset


  


    

        
    
    

    
 

    