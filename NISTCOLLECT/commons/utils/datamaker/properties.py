import asyncio
import concurrent.futures
import pandas as pd
import pubchempy as pcp
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor

from NISTCOLLECT.commons.constants import CONFIG, DATA_PATH
from NISTCOLLECT.commons.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class MolecularProperties:    
    
    def __init__(self):
        pass  

    #--------------------------------------------------------------------------
    def get_properties_from_single_name(self, data):

        name = data.get('name', '')
        try:
            cid = pcp.get_cids(name, list_return='flat')
            properties = pcp.Compound.from_cid(cid).to_dict()
            logger.debug(f'Successfully retrieved properties for {name}')
            return properties
        except Exception as e:
            logger.error(f'Error fetching properties for {name}: {e}')
            return {}

    #--------------------------------------------------------------------------
    def get_properties_from_multiple_names(self, data):

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_name = {executor.submit(self.get_properties_from_single_name, dt): dt for dt in data}
            results = {}
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    logger.error(f'Error processing {name}: {e}')
                    results[name] = {}
        return results
