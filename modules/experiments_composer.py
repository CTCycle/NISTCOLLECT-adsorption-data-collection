import os
import pandas as pd
import boto3
from io import StringIO

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_assets import AdsorptionDataset
from modules.components.scraper_assets import NISTAdsorptionAPI
import modules.global_variables as GlobVar
import configurations as cnf

# [BUILD ADSORPTION EXPERIMENTS DATASET]
#==============================================================================
# Builds the index of adsorption experiments as from the NIST database.
# such index will be used to extract single experiment adsorption data 
# split the dataset into experiments with a single component and with binary mixtures.
# extract the adsorption data embedded in the NIST json dictionaries and add them to
# custom columns for pressure and uptake. Eventually, explode the dataset to ensure
# each row will contain a specific pair of pressure-uptake data points.
#==============================================================================
print('''
-------------------------------------------------------------------------------
COLLECT ADSORPTION DATA
-------------------------------------------------------------------------------
''')

webworker = NISTAdsorptionAPI()

# get isothernm experiments index
#------------------------------------------------------------------------------
isotherm_index = webworker.Get_Isotherms_Index()
isotherm_index = isotherm_index[:int(len(isotherm_index) * cnf.experiments_fraction)] 
isotherm_names = [x['filename'] for x in isotherm_index]
df_experiments = pd.DataFrame(isotherm_index)    
print(f'Total number of adsorption experiments: {df_experiments.shape[0]}')
print()

# function to split list of names into chunks to avoid out of memory issues
#------------------------------------------------------------------------------
window_size = int(cnf.chunk_size * len(isotherm_names))
def list_fragmenter(lst, n):    
    for i in range(0, len(lst), n):
        yield lst[i:i + n]    

# collect actual adsorption data using the experiments index,
# split dataset based on single or binary mixture composition
# extract experimental data from the datasets and expand the latter
# save files in s3 bucket or locally
#------------------------------------------------------------------------------
drop_columns = ['category', 'tabular_data', 'isotherm_type', 'digitizer', 'articleSource']
for i, fg in enumerate(list_fragmenter(isotherm_names, window_size)):
    isotherm_data = webworker.Get_Isotherms_Data(fg)
    df_isotherms = pd.DataFrame(isotherm_data).drop(columns = drop_columns)     
    dataworker = AdsorptionDataset(df_isotherms)
    single_compound, binary_mixture = dataworker.split_by_mixcomplexity()
    SC_dataset = dataworker.extract_adsorption_data(single_compound, num_species=1) 
    BM_dataset = dataworker.extract_adsorption_data(binary_mixture, num_species=2) 
    SC_dataset_expanded, BM_dataset_expanded = dataworker.dataset_expansion(SC_dataset, BM_dataset) 
    if cnf.output_type == 'HOST':            
        file_loc = os.path.join(GlobVar.data_path, 'single_component_dataset.csv') 
        SC_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index = False, sep = ';', encoding='utf-8')
        file_loc = os.path.join(GlobVar.data_path, 'binary_mixture_dataset.csv') 
        BM_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index = False, sep = ';', encoding='utf-8')
    else:
        s3_resource = boto3.resource('s3', region_name=cnf.region_name)        
        csv_buffer = StringIO()
        SC_dataset_expanded.to_csv(csv_buffer)    
        s3_resource.Object(cnf.S3_bucket_name, 'single_component_dataset.csv').put(Body=csv_buffer.getvalue())
        csv_buffer = StringIO()
        BM_dataset_expanded.to_csv(csv_buffer)    
        s3_resource.Object(cnf.S3_bucket_name, 'binary_mixture_dataset.csv').put(Body=csv_buffer.getvalue())
   
print('''
-------------------------------------------------------------------------------
NISTADS data collection has terminated. All files have been saved and are ready
to be used! Enjoy your data 
-------------------------------------------------------------------------------
''')
