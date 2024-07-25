import asyncio
import aiohttp
import pandas as pd
import pubchempy as pcp

from NISTCOLLECT.commons.utils.datascraper.asynchronous import properties_from_multiple_names
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

        names = [x.get('name', 'NA') for x in data]
        loop = asyncio.get_event_loop()
        extracted_properties = loop.run_until_complete(properties_from_multiple_names(names))

        
        #df_properties = pd.DataFrame(extracted_properties)

        return extracted_properties

        
             
         

