import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
tqdm.pandas()

from NISTCOLLECT.commons.constants import CONFIG, DATA_PATH
from NISTCOLLECT.commons.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class MolecularProperties:    
    
    def __init__(self):
        pass   

    #--------------------------------------------------------------------------
    def get_properties_by_name(self, data):  
        compound_name = data.get('name', '') 
        try:           
            cid = pcp.get_cids(compound_name, list_return='flat') 
            properties = pcp.Compound.from_cid(cid).to_dict()
            logger.debug(f'Successfully retrieved properties for {compound_name}')
        except:
            logger.error(f'Could not find molecular properties of {compound_name}')
            properties = {}

        return properties
    
    #--------------------------------------------------------------------------
    def extract_molecular_properties(self, data):

        # create list of molecular properties of sorbates (using PUG REST API as reference)
        extracted_properties = [self.get_properties_by_name(d) for d in data if d is not None]        
        # create dataframe with data extracted
        df_properties = pd.DataFrame(extracted_properties)

        return df_properties

        
             
         

