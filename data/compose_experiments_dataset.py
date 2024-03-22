import os
import sys
import pandas as pd
import boto3
from io import StringIO

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and classes
#------------------------------------------------------------------------------
from utils.data_assets import AdsorptionDataset
from utils.scraper_assets import NISTAdsorptionAPI
import utils.global_paths as globpt
import configurations as cnf

# [BUILD ADSORPTION EXPERIMENTS DATASET]
#==============================================================================
# Builds the index of adsorption experiments from the NIST database data, which
# will be used to extract adsorption experiments data 
#==============================================================================
print('\nCollect experiments data\n')

# get isotherm experiments index using the NIST API
#------------------------------------------------------------------------------
webworker = NISTAdsorptionAPI()
isotherm_index = webworker.Get_Isotherms_Index()
isotherm_index = isotherm_index[:int(len(isotherm_index) * cnf.experiments_fraction)] 
isotherm_names = [x['filename'] for x in isotherm_index]

# create a dataframe with extracted experiments data
#-----------------------------------------------------------------------------
df_experiments = pd.DataFrame(isotherm_index)    
print(f'Total number of adsorption experiments: {df_experiments.shape[0]}')
print()

# define a function to split the index into chunks to be processed sequentially,
# to avoid excessive burden on system memory (for low RAM machines)
#------------------------------------------------------------------------------
window_size = int(cnf.chunk_size * len(isotherm_names))
def list_fragmenter(lst, n):    
    for i in range(0, len(lst), n):
        yield lst[i:i + n]    

# collect actual adsorption data using the list of index chunks 
#------------------------------------------------------------------------------
drop_columns = ['category', 'tabular_data', 'isotherm_type', 'digitizer', 'articleSource']
for i, fg in enumerate(list_fragmenter(isotherm_names, window_size)):
    isotherm_data = webworker.Get_Isotherms_Data(fg)
    df_isotherms = pd.DataFrame(isotherm_data).drop(columns = drop_columns)     
    dataworker = AdsorptionDataset(df_isotherms)

    # split the chunk dataset based on whether experiments is performed on single 
    # components or binary mixtures
    single_compound, binary_mixture = dataworker.split_by_mixcomplexity()

    # extracts the adsorption experiments data and expand the dataset
    SC_dataset = dataworker.extract_adsorption_data(single_compound, num_species=1) 
    BM_dataset = dataworker.extract_adsorption_data(binary_mixture, num_species=2)

    # save data either locally or in a S3 bucket as .csv files
    SC_dataset_expanded, BM_dataset_expanded = dataworker.dataset_expansion(SC_dataset, BM_dataset) 
    file_loc = os.path.join(globpt.data_path, 'single_component_dataset.csv') 
    SC_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index = False, sep = ';', encoding='utf-8')
    file_loc = os.path.join(globpt.data_path, 'binary_mixture_dataset.csv') 
    BM_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index = False, sep = ';', encoding='utf-8')

   
print('''
-------------------------------------------------------------------------------
NISTADS data collection has terminated. All files have been saved and are ready
to be used! Enjoy your data 
-------------------------------------------------------------------------------
''')
