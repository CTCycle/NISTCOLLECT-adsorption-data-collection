import pandas as pd
from tqdm import tqdm
tqdm.pandas()      


# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values 
###############################################################################
def filter_outside_boundaries(dataset : pd.DataFrame, max_pressure, max_uptake):

    dataset = dataset[dataset['temperature'].astype(int) > 0]
    dataset = dataset[dataset['pressure_in_Pascal' ].astype(float).between(0.0, max_pressure)]
    dataset = dataset[dataset['uptake_in_mol_g'].astype(float).between(0.0, max_uptake)] 

    return dataset


  


    

        
    
    

    
 

    