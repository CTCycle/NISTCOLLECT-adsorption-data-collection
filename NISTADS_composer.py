import os
import art
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import boto3
from io import StringIO

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_classes import AdsorptionDataset, DataStorage
from modules.components.scraper_classes import NISTAdsorptionAPI
import modules.global_variables as GlobVar
import modules.configurations as cnf

# welcome message
#------------------------------------------------------------------------------
ascii_art = art.text2art('NIST data collection')
print(ascii_art)

# [BUILD ADSORBATES AND ADSORBENTS DATASETS]
#==============================================================================
# module for the selection of different operations
#==============================================================================
print('''
-------------------------------------------------------------------------------
BUILD ADSORBENT/ADSORBATES DATASETS
-------------------------------------------------------------------------------
''')

webworker = NISTAdsorptionAPI()

# get index of guests and hosts index and names from json file
#------------------------------------------------------------------------------
adsorbates_index, adsorbents_index = webworker.Get_GuestHost_Index()
adsorbates_index = adsorbates_index[:int(len(adsorbates_index) * cnf.guest_fraction)] 
adsorbents_index = adsorbents_index[:int(len(adsorbents_index) * cnf.host_fraction)]
adsorbates_names = [x['InChIKey'] for x in adsorbates_index]
adsorbents_names = [x['hashkey'] for x in adsorbents_index]

# create dataset from guest and host indexes
#------------------------------------------------------------------------------
df_adsorbates = pd.DataFrame(adsorbates_index)
df_adsorbents = pd.DataFrame(adsorbents_index) 
print(f'''
Total number of adsorbents: {df_adsorbents.shape[0]}
Total number of adsorbates: {df_adsorbates.shape[0]}
''')

# extract data for the adsorbents based on previously extracted indexes
#------------------------------------------------------------------------------
print('Extracting adsorbents data...')
df_adsorbents_properties = webworker.Get_GuestHost_Data(adsorbents_names, focus = 'host')    
df_hosts = pd.DataFrame(df_adsorbents_properties) 
print()

# extract data for the sorbates based on previously extracted indexes
#------------------------------------------------------------------------------
print('Extracting adsorbates data...')
df_adsorbates_properties = webworker.Get_GuestHost_Data(adsorbates_names, focus = 'guest')    
df_guests = pd.DataFrame(df_adsorbates_properties)
print()

# create list of molecular properties of sorbates (using pubchem as reference)
#------------------------------------------------------------------------------
print('Adding physicochemical properties to guest molecules dataset...')
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
print()

# extract single properties from the general list and create a dictionary with
# property names and values
#------------------------------------------------------------------------------
canonical_smiles = [x['canonical_smiles'] if x != 'None' else 'NaN' for x in adsorbates_properties]
complexity = [x['complexity'] if x != 'None' else 'NaN' for x in adsorbates_properties]
atoms = [' '.join(x['elements']) if x != 'None' else 'NaN' for x in adsorbates_properties]
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

# save files either as csv locally or in S3 bucket
#------------------------------------------------------------------------------
datastorage = DataStorage()
if cnf.output_type == 'HOST':
    file_loc = os.path.join(GlobVar.data_path, 'adsorbents_dataset.csv') 
    df_hosts.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
    file_loc = os.path.join(GlobVar.data_path, 'adsorbates_dataset.csv') 
    df_guests_expanded.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')    
else:
    s3_resource = boto3.resource('s3', region_name=cnf.region_name)
    csv_buffer = StringIO()
    df_hosts.to_csv(csv_buffer)    
    s3_resource.Object(cnf.S3_bucket_name, 'adsorbents_dataset.csv').put(Body=csv_buffer.getvalue())
    csv_buffer = StringIO()
    df_guests_expanded.to_csv(csv_buffer)    
    s3_resource.Object(cnf.S3_bucket_name, 'adsorbates_dataset.csv').put(Body=csv_buffer.getvalue())
    
# [BUILD ADSORPTION EXPERIMENTS DATASET]
# [EXTRACT ADSORPTION DATA AND FILTER DATASET BASED ON ADSORPTION UNITS]
#==============================================================================
# Builds the index of adsorption experiments as from the NIST database.
# such index will be used to extract single experiment adsorption data 
#==============================================================================
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
datastorage = DataStorage()
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
