import os
import pandas as pd
import sys
import pubchempy as pcp
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.components.data_classes import AdsorptionDataset
from modules.components.scraper_classes import NISTAdsorptionAPI
import modules.global_variables as GlobVar

# [BUILD ADSORBATES AND ADSORBENTS DATASETS]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print(f'''
-------------------------------------------------------------------------------
BUILD DATASETS
-------------------------------------------------------------------------------
...
''')

print('''STEP 1 ---> BUILD ADSORBENT/ADSORBATES DATASETS
''')

webworker = NISTAdsorptionAPI()

# get index of guests and hosts from json file
#------------------------------------------------------------------------------
adsorbates_index, adsorbents_index = webworker.Get_GuestHost_Index()
adsorbates_names, adsorbents_names = webworker.Get_GuestHost_Names()
df_adsorbates = pd.DataFrame(adsorbates_index)
df_adsorbents = pd.DataFrame(adsorbents_index) 
print(f'''Total number of adsorbents: {df_adsorbents.shape[0]}
''')

# extract data for the adsorbents based on previously extracted indexes
#------------------------------------------------------------------------------
df_adsorbents_properties = webworker.Get_GuestHost_Data(focus = 'host')    
df_hosts = pd.DataFrame(df_adsorbents_properties)
print()  

# extract data for the sorbates based on previously extracted indexes
#------------------------------------------------------------------------------
print(f'''Total number of adsorbates: {df_adsorbates.shape[0]}
''')
df_adsorbates_properties = webworker.Get_GuestHost_Data(focus = 'guest')    
df_guests = pd.DataFrame(df_adsorbates_properties)

# create list of molecular properties of sorbates (using pubchem as reference)
#------------------------------------------------------------------------------
adsorbates_properties = []
for row in tqdm(df_adsorbates.itertuples(), total = len(df_adsorbates)):    
    name = row[2].lower()        
    try:
        cid = pcp.get_cids(name, list_return='flat')
        properties = pcp.Compound.from_cid(cid).to_dict()
        adsorbates_properties.append(properties)
    except:
        properties = 'None'
        adsorbates_properties.append(properties)    

# extract single properties from the general list and create a dictionary with
# property names and values
#------------------------------------------------------------------------------
canonical_smiles = [x['canonical_smiles'] if x != 'None' else 'NaN' for x in adsorbates_properties]
complexity = [x['complexity'] if x != 'None' else 'NaN' for x in adsorbates_properties]
atoms = [x['elements'] if x != 'None' else 'NaN' for x in adsorbates_properties]
mol_weight = [x['molecular_weight'] if x != 'None' else 'NaN' for x in adsorbates_properties]
covalent_units = [x['covalent_unit_count'] if x != 'None' else 'NaN' for x in adsorbates_properties]
H_acceptors = [x['h_bond_acceptor_count'] if x != 'None' else 'NaN' for x in adsorbates_properties]
H_donors = [x['h_bond_donor_count'] if x != 'None' else 'NaN' for x in adsorbates_properties]
heavy_atoms = [x['heavy_atom_count'] if x != 'None' else 'NaN' for x in adsorbates_properties]

properties = {'canonical_smiles': canonical_smiles,
              'complexity': complexity,
              'atoms': atoms,
              'mol_weight': mol_weight,
              'covalent_units': covalent_units,
              'H_acceptors': H_acceptors,
              'H_donors': H_donors,
              'heavy_atoms': heavy_atoms}

# create dataset of properties and concatenate it with sorbates dataset
#------------------------------------------------------------------------------
df_properties = pd.DataFrame(properties)
df_guests_expanded = pd.concat([df_guests, df_properties], axis = 1)

# save files as csv
#------------------------------------------------------------------------------
file_loc = os.path.join(GlobVar.data_path, 'adsorbents_dataset.csv') 
df_hosts.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
file_loc = os.path.join(GlobVar.data_path, 'adsorbates_dataset.csv') 
df_guests_expanded.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8') 

# [BUILD ADSORPTION EXPERIMENTS DATASET]
#==============================================================================
# Builds the index of adsorption experiments as from the NIST database.
# such index will be used to extract single experiment adsorption data 
#==============================================================================
print('''STEP 2 ---> COLLECT ADSORPTION DATA INDEX
''')

webworker = NISTAdsorptionAPI()
isotherm_index = webworker.Get_Isotherms_Index()
isotherm_names = webworker.Get_Isotherms_Names()
df_experiments = pd.DataFrame(isotherm_index)    
print('Total number of adsorption experiments: {}'.format(df_experiments.shape[0]))
print()
    
# collect actual adsorption data using the experiments index
#------------------------------------------------------------------------------
isotherm_data = webworker.Get_Isotherms_Data()
df_isotherms = pd.DataFrame(isotherm_data)
drop_columns = ['category', 'tabular_data', 'isotherm_type', 'digitizer', 'articleSource']
df_isotherms = df_isotherms.drop(columns = drop_columns)
print() 
    
# [EXTRACT ADSORPTION DATA AND FILTER DATASET BASED ON ADSORPTION UNITS]
#==============================================================================
# split the dataset into experiments with a single component and with binary mixtures.
# extract the adsorption data embedded in the NIST json dictionaries and add them to
# custom columns for pressure and uptake. Eventually, explode the dataset to ensure
# each row will contain a specific pair of pressure-uptake data points.
#==============================================================================
print('''STEP 3 ---> PREPARING PRELIMINARY VERSION OF ADSORPTIONN DATASET
''')

dataworker = AdsorptionDataset(df_isotherms)

# split dataset based on single or binary mixture composition
#------------------------------------------------------------------------------
grouped_datasets = dataworker.split_by_mixcomplexity()

# split dataset based on single or binary mixture composition
#------------------------------------------------------------------------------
SC_dataset, BM_dataset = dataworker.extract_adsorption_data()  
SC_dataset_expanded, BM_dataset_expanded = dataworker.dataset_expansion()    
print()  
    
# [SAVING DATASET INTO FILES]
#==============================================================================
# save files in the dataset folder as .csv files
#==============================================================================
print('STEP 4 ----> Saving files')
print()
file_loc = os.path.join(GlobVar.data_path, 'single_component_dataset.csv') 
SC_dataset_expanded.to_csv(file_loc, index = False, sep = ';', encoding='utf-8')
file_loc = os.path.join(GlobVar.data_path, 'binary_mixture_dataset.csv') 
BM_dataset_expanded.to_csv(file_loc, index = False, sep = ';', encoding='utf-8')
  






