import pandas as pd
from tqdm import tqdm
import requests as r
import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio

from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# function to retrieve HTML data
###############################################################################
async def fetch_from_single_URL(self, session, url, semaphore):
    async with semaphore:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f'Could not fetch data from {url}. Status: {response.status}')
                return None
            try:
                return await response.json()
            except aiohttp.client_exceptions.ContentTypeError as e:
                logger.error(f'Error decoding JSON from {url}: {e}')
                return None

# function to retrieve HTML data
###############################################################################
async def fetch_data_chunk(self, urls):
    semaphore = asyncio.Semaphore(self.max_parallel_calls)
    async with aiohttp.ClientSession() as session:
        tasks = [self.fetch_from_single_URL(session, url, semaphore) for url in urls]
        results = await tqdm_asyncio.gather(*tasks)
    return results


# [NIST DATABASE API: GUEST/HOST]
###############################################################################
class GuestHostAPI: 

    def __init__(self):      
        self.url_GUEST = 'https://adsorption.nist.gov/isodb/api/gases.json'
        self.url_HOST = 'https://adsorption.nist.gov/matdb/api/materials.json'
        self.guest_fraction = CONFIG["GUEST_FRACTION"]
        self.host_fraction = CONFIG["HOST_FRACTION"]
        self.guest_identifier = 'InChIKey'
        self.host_identifier = 'hashkey'
        self.max_parallel_calls = CONFIG["PARALLEL_TASKS"]

    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_guest_host_indexes(self):
        
        '''
        Retrieves adsorbates and adsorbents data from specified URLs. This function sends GET 
        requests to the URLs specified in the instance variables `self.url_adsorbents` and `self.url_adsorbates`. 
        It then checks the status of the response and if successful (status code 200), 
        it converts the JSON response to a Python dictionary. If the request fails, 
        it prints an error message and sets the corresponding index to None.

        Returns:            
            tuple: A tuple containing two elements:
                - adsorbates_index (dict or None): A dictionary containing the adsorbates data if the request was successful, None otherwise.
                - adsorbents_index (dict or None): A dictionary containing the adsorbents data if the request was successful, None otherwise.
        
        ''' 
        guest_json = r.get(self.url_GUEST)
        host_json = r.get(self.url_HOST)

        if guest_json.status_code == 200:
            guest_index = guest_json.json() 
            df_guest = pd.DataFrame(guest_index)
            logger.info(f'Total number of adsorbents: {df_guest.shape[0]}')
        else:
            logger.error(f'Failed to retrieve adsorbents data. Status code: {guest_json.status_code}')       
            df_guest = None
        if host_json.status_code == 200:
            host_index = host_json.json() 
            df_host = pd.DataFrame(host_index) 
            logger.info(f'Total number of adsorbates: {df_host.shape[0]}')
        else:
            logger.error(f'Failed to retrieve adsorbates data. Status code: {host_json.status_code}')
            df_host = None
  
        return df_guest, df_host     

    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_guest_host_data(self, df_guest, df_host):

        loop = asyncio.get_event_loop()

        if df_guest is not None:
            guest_names = df_guest[self.guest_identifier].to_list()[:100]
            guest_urls = [f'https://adsorption.nist.gov/isodb/api/gas/{name}.json' for name in guest_names]
            guest_data = loop.run_until_complete(fetch_data_chunk(guest_urls))
            guest_data = [data for data in guest_data if data is not None]
        else:
            logger.error('No available guest data has been found. Skipping directly to host species')
            guest_data = None

        if df_host is not None:  
            host_names = df_host[self.host_identifier].to_list()[:100]        
            host_urls = [f'https://adsorption.nist.gov/isodb/api/material/{name}.json' for name in host_names]       
            host_data = loop.run_until_complete(fetch_data_chunk(host_urls))        
            host_data = [data for data in host_data if data is not None]
        else:
            logger.error('No available host data has been found.')
            host_data = None

        return guest_data, host_data    



# [NIST DATABASE API]
###############################################################################
class NISTAdsorptionAPI:  
    
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
    
    
            
        
            
       
            
                
                
