import os
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class MolecularProperties:    
    
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
    def get_properties_from_single_name(self, name):
        
        try:
            compounds = pcp.get_compounds(name.lower(), namespace='name', list_return='flat')
            properties = compounds[0].to_dict()
            logger.debug(f'Successfully retrieved properties for {name}')
            return properties
        except Exception as e:
            logger.error(f'Error fetching properties for {name}: {e}')
            return {}

    #--------------------------------------------------------------------------
    def get_properties_from_multiple_names(self, data):

        names = (dt.get('name', '') for dt in data)        
        for name in tqdm(names, total=len(data)):
            features = self.get_properties_from_single_name(name)
            all_properties = self.process_extracted_properties(name, features)           

        self.save_properties_dataframe(all_properties)     

        return all_properties
    
    #--------------------------------------------------------------------------
    def process_extracted_properties(self, name, features):
           
        self.properties['name'].append(name)
        self.properties['atoms'].append(features['atoms'])
        self.properties['heavy_atoms'].append(features['heavy_atom_count'])
        self.properties['bonds'].append(features['bonds'])
        self.properties['elements'].append(' '.join(features['elements']))
        self.properties['molecular_weight'].append(features['molecular_weight'])
        self.properties['molecular_formula'].append(features['molecular_formula'])
        self.properties['SMILE'].append(features['canonical_smiles'])
        self.properties['H_acceptors'].append(features['h_bond_acceptor_count'])
        self.properties['H_donors'].append(features['h_bond_donor_count'])          
        
        return self.properties    
        
    #--------------------------------------------------------------------------
    def save_properties_dataframe(self, properties):

        dataframe = pd.DataFrame(properties)  
        file_loc = os.path.join(DATA_MAT_PATH, 'guests_dataset.csv') 
        dataframe.to_csv(file_loc, index=False, sep=';', encoding='utf-8')       
