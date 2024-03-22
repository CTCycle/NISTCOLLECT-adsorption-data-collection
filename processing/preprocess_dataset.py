import os
import sys
import pandas as pd

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and classes
#------------------------------------------------------------------------------
from utils.data_assets import PreProcessing
import utils.global_paths as globpt
import configurations as cnf

# specify relative paths from global paths and create subfolders
#------------------------------------------------------------------------------
mat_path = os.path.join(globpt.data_path, 'materials') 
exp_path = os.path.join(globpt.data_path, 'experiments') 
os.mkdir(mat_path) if not os.path.exists(mat_path) else None
os.mkdir(exp_path) if not os.path.exists(exp_path) else None 

# [LOAD DATA]
#==============================================================================
#==============================================================================
filepath = os.path.join(mat_path, 'adsorbents_dataset.csv')  
df_adsorbents = pd.read_csv(filepath, sep=';', encoding='utf-8')  
filepath = os.path.join(mat_path, 'adsorbates_dataset.csv')  
df_adsorbates = pd.read_csv(filepath, sep=';', encoding='utf-8')  
filepath = os.path.join(exp_path, 'single_component_dataset.csv')  
df_SC = pd.read_csv(filepath, sep=';', encoding='utf-8')

# [PREPROCESS DATA]
#==============================================================================
#==============================================================================

preprocessor = PreProcessing()

# add molecular properties based on PUGCHEM API data
#------------------------------------------------------------------------------ 
print('\nAdding physicochemical properties from guest species dataset')
dataset = preprocessor.add_guest_properties(df_SC, df_adsorbates)
dataset = dataset.dropna()

# filter experiments leaving only valid uptake and pressure units, then convert 
# pressure and uptake to Pa (pressure) and mol/kg (uptake)
#------------------------------------------------------------------------------
print('\nConverting units and filtering bad values\n')

# filter experiments by pressure and uptake units 
dataset = dataset[dataset[preprocessor.Q_unit_col].isin(preprocessor.valid_units)]


# convert pressures to Pascal
dataset[preprocessor.P_col] = dataset.progress_apply(lambda x : 
                                                     preprocessor.pressure_converter(x[preprocessor.P_unit_col], 
                                                                                                x['pressure']), 
                                                                                                axis = 1)
# convert uptakes to mol/g
dataset[preprocessor.Q_col] = dataset.progress_apply(lambda x : 
                                                     preprocessor.uptake_converter(x[preprocessor.Q_unit_col], 
                                                                                              x['adsorbed_amount'], 
                                                                                              x['mol_weight']), 
                                                                                              axis = 1)

# further filter the dataset to remove experiments which values are outside desired boundaries, 
# such as experiments with negative temperature, pressure and uptake values
#------------------------------------------------------------------------------ 
dataset = dataset[dataset['temperature'].astype(int) > 0]
dataset = dataset[dataset[preprocessor.P_col].astype(float).between(0.0, cnf.max_pressure)]
dataset = dataset[dataset[preprocessor.Q_col].astype(float).between(0.0, cnf.max_uptake)]

# Aggregate values using groupby function in order to group the dataset by experiments
#------------------------------------------------------------------------------ 
def join_str(x):
    return ' '.join(x.astype(str))

aggregate_dict = {'temperature' : 'first',                  
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
   
# group dataset by experiments and drop filename column as it is not necessary
print('\nGroup by experiment and clean data\n')
dataset_grouped = dataset.groupby('filename', as_index=False).agg(aggregate_dict)
dataset_grouped.drop(columns='filename', axis=1, inplace=True)

# remove series of pressure/uptake with less than X points, drop rows containing nan
# values and select a subset of samples for training
#------------------------------------------------------------------------------ 
dataset_grouped = dataset_grouped[~dataset_grouped[preprocessor.P_col].apply(lambda x: all(elem == 0 for elem in x))]
dataset_grouped = dataset_grouped[dataset_grouped[preprocessor.P_col].apply(lambda x: len(x)) >= cnf.min_points]
dataset_grouped = dataset_grouped.dropna()
total_experiments = dataset_grouped.shape[0]

# preprocess sequences to remove leading 0 values (some experiments may have several
# zero measurements at the start), make sure that every experiment starts with pressure
# of 0 Pa and uptake of 0 mol/g (effectively converges to zero)
#------------------------------------------------------------------------------
dataset_grouped[[preprocessor.P_col, preprocessor.Q_col]] = dataset_grouped.apply(lambda row: 
                 preprocessor.remove_leading_zeros(row[preprocessor.P_col],
                 row[preprocessor.Q_col]), axis=1, result_type='expand')

# save files as csv locally
#------------------------------------------------------------------------------
file_loc = os.path.join(exp_path, 'preprocessed_SC_dataset.csv') 
dataset_grouped.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
 

