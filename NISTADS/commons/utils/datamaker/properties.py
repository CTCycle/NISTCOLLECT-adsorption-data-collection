import os
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger

 


# [DATASET OPERATIONS]
###############################################################################
class GuestProperties:    
    
    def __init__(self):

        self.properties = {'name' : [],
                            'atoms' : [],
                            'heavy_atoms' : [],
                            'bonds' : [],
                            'elements' : [],
                            'molecular_weight' : [],
                            'molecular_formula' : [],
                            'SMILE' : [],
                            'H_acceptors' : [],
                            'H_donors' : [],
                            'heavy_atoms' : []}

    
    #--------------------------------------------------------------------------
    def get_properties_for_single_guest(self, name):
        
        try:
            compounds = pcp.get_compounds(name.lower(), namespace='name', list_return='flat')
            properties = compounds[0].to_dict()
            logger.debug(f'Successfully retrieved properties for {name}')
            return properties
        except Exception as e:
            logger.error(f'Error fetching properties for {name}: {e}')
            return {}

    #--------------------------------------------------------------------------
    def get_properties_for_multiple_guests(self, data):

        names = (dt.get('name', '') for dt in data)        
        for name in tqdm(names, total=len(data)):
            features = self.get_properties_for_single_guest(name)
            all_properties = self.process_extracted_properties(name, features)             

        return all_properties
    
    #--------------------------------------------------------------------------
    def process_extracted_properties(self, name, features):
           
        self.properties['name'].append(name)
        self.properties['atoms'].append(features.get('atoms', 'NA'))
        self.properties['heavy_atoms'].append(features.get('heavy_atom_count', 'NA'))
        self.properties['bonds'].append(features.get('bonds', 'NA'))
        self.properties['elements'].append(' '.join(features.get('elements', 'NA')))
        self.properties['molecular_weight'].append(features.get('molecular_weight', 'NA'))
        self.properties['molecular_formula'].append(features.get('molecular_formula', 'NA'))
        self.properties['SMILE'].append(features.get('canonical_smiles', 'NA'))
        self.properties['H_acceptors'].append(features.get('h_bond_acceptor_count', 'NA'))
        self.properties['H_donors'].append(features.get('h_bond_donor_count', 'NA'))          
        
        return self.properties    
        
    #--------------------------------------------------------------------------
    def save_properties_dataframe(self, guest_properties=None):

        if guest_properties is not None:            
            guest_data = pd.DataFrame(guest_properties)
            file_loc = os.path.join(DATA_PATH, 'guests_dataset.csv') 
            guest_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')         


# [DATASET OPERATIONS]
###############################################################################
class HostProperties:    
    
    def __init__(self):
        pass   
    
        
    #--------------------------------------------------------------------------
    def save_properties_dataframe(self, host_properties=None):
        
        if host_properties is not None:             
            host_data = pd.DataFrame(host_properties)             
            file_loc = os.path.join(DATA_PATH, 'host_dataset.csv') 
            host_data.to_csv(file_loc, index=False, sep=';', encoding='utf-8')   




# [DATASET OPERATIONS]
###############################################################################
class FetchMolecularProperties: 

    def __init__(self):
        self.guest_property = GuestProperties()

    def get_guest_properties(self, data):
    
        guest_properties = self.guest_property.get_properties_for_multiple_guests(data)