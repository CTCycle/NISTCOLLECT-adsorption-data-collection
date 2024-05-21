import os
import sys
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [DEFINE PROJECT FOLDER PATH]
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir) 

# [IMPORT CUSTOM MODULES]
from utils.API.core import NISTAdsorptionAPI
from config.pathfinder import DATA_MAT_PATH
import config.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [GET ISOTHERM MATERIALS INDEX]
    #--------------------------------------------------------------------------
    print('\nCollect experiments data\n')

    # get isotherm indexes invoking API
    webworker = NISTAdsorptionAPI()
    adsorbates_index, adsorbents_index = webworker.Get_GuestHost_Index()
    adsorbates_index = adsorbates_index[:int(len(adsorbates_index) * cnf.GUEST_FRACTION)] 
    adsorbents_index = adsorbents_index[:int(len(adsorbents_index) * cnf.HOST_FRACTION)]
    adsorbates_names = [x['InChIKey'] for x in adsorbates_index]
    adsorbents_names = [x['hashkey'] for x in adsorbents_index]

    # create a dataframe with extracted experiments data 
    df_adsorbates = pd.DataFrame(adsorbates_index)
    df_adsorbents = pd.DataFrame(adsorbents_index) 
    print(f'Total number of adsorbents: {df_adsorbents.shape[0]}')
    print(f'Total number of adsorbates: {df_adsorbates.shape[0]}')

    # 2. [COLLECT HOST DATA]
    #--------------------------------------------------------------------------
    print('\nExtracting adsorbents data')
    df_adsorbents_properties = webworker.Get_GuestHost_Data(adsorbents_names, focus = 'host')    
    df_hosts = pd.DataFrame(df_adsorbents_properties) 
    
    # 3. [COLLECT GUEST DATA]
    #--------------------------------------------------------------------------
    print('\nExtracting adsorbates data')
    df_adsorbates_properties = webworker.Get_GuestHost_Data(adsorbates_names, focus = 'guest')    
    df_guests = pd.DataFrame(df_adsorbates_properties)
    
    # create list of molecular properties of sorbates (using PUG REST API as reference)    
    print('\nAdding physicochemical properties to guest molecules dataset')
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
    df_properties = pd.DataFrame(properties)
    df_guests_expanded = pd.concat([df_guests, df_properties], axis = 1)

    # save files either as csv locally or in S3 bucket    
    file_loc = os.path.join(DATA_MAT_PATH, 'adsorbents_dataset.csv') 
    df_hosts.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
    file_loc = os.path.join(DATA_MAT_PATH, 'adsorbates_dataset.csv') 
    df_guests_expanded.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')    

    print('NISTADS data collection has terminated. All files have been saved.')