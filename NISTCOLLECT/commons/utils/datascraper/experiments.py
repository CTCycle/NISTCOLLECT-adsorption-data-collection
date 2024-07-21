import pandas as pd
from tqdm import tqdm
import requests as r
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio

from NISTCOLLECT.commons.utils.datascraper.status import GetServerStatus
from NISTCOLLECT.commons.utils.datascraper.asynchronous import fetch_data_chunk
from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# [CHECK SERVER STATUS]
###############################################################################
server = GetServerStatus()
server.check_status()


# [NIST DATABASE API]
###############################################################################
class AdsorptionDataAPI:  
    
    def __init__(self):        
        self.url_isotherms = 'https://adsorption.nist.gov/isodb/api/isotherms.json'
    
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_experiments_index(self):
        
        '''
        This function retrieves the index of isotherms from the NIST ISODB by 
        sending a GET request to the database. The returned JSON data is returned as is.
        If the request fails, an error message is printed and None is returned.        

        Returns:
            isotherm_index (dict or None): A dictionary containing the isotherm index 
                                           if the request was successful, None otherwise.       
            
        ''' 
        response = r.get(self.url_isotherms)
        if response.status_code == 200:             
            isotherm_index = response.json()    
            logger.info('Successfully retrieve adsorption isotherm index')    
        else:
            logger.error(f'Error: Failed to retrieve data. Status code: {response.status_code}')
            isotherm_index = None
            
        return isotherm_index    
    
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_experiments_data(self, exp_names): 

        '''        
        Retrieve isotherm data for a given list of experiment names from the NIST ISODB.
        
        This function sends a GET request to the NIST ISODB for each experiment name in the list,
        retrieves the corresponding isotherm data in JSON format, and appends the data to a list. 
        It uses asynchronous operations to fetch the data efficiently.

        Keyword arguments:
            exp_names (list): A list of experiment names for which isotherm data is to be retrieved.

        Returns:
            list: A list of dictionaries where each dictionary contains the isotherm data for an experiment.

        '''  
        loop = asyncio.get_event_loop()
        exp_by_url = [f'https://adsorption.nist.gov/isodb/api/isotherm/{name}.json' for name in exp_names]
        exp_data = loop.run_until_complete(fetch_data_chunk(exp_by_url))
        exp_data = [data for data in exp_data if data is not None]                  
                
        return exp_data
    
    
            
        
            
       
            
                
                
