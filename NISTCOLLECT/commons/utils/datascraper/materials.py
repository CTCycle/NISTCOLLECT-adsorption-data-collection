import numpy as np
import pandas as pd
import requests as r
import asyncio

from NISTCOLLECT.commons.utils.datascraper.status import GetServerStatus
from NISTCOLLECT.commons.utils.datascraper.asynchronous import data_from_multiple_URLs
from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# [CHECK SERVER STATUS]
###############################################################################
server = GetServerStatus()
server.check_status()


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

        guest_samples = int(np.ceil(CONFIG["GUEST_FRACTION"] * df_guest.shape[0]))
        host_samples = int(np.ceil(CONFIG["HOST_FRACTION"] * df_host.shape[0]))
        loop = asyncio.get_event_loop()

        if df_guest is not None:
            guest_names = df_guest[self.guest_identifier].to_list()[:guest_samples]
            guest_urls = [f'https://adsorption.nist.gov/isodb/api/gas/{name}.json' for name in guest_names]
            guest_data = loop.run_until_complete(data_from_multiple_URLs(guest_urls))
            guest_data = [data for data in guest_data if data is not None]
        else:
            logger.error('No available guest data has been found. Skipping directly to host species')
            guest_data = None

        if df_host is not None:  
            host_names = df_host[self.host_identifier].to_list()[:host_samples]       
            host_urls = [f'https://adsorption.nist.gov/isodb/api/material/{name}.json' for name in host_names]       
            host_data = loop.run_until_complete(data_from_multiple_URLs(host_urls))        
            host_data = [data for data in host_data if data is not None]
        else:
            logger.error('No available host data has been found.')
            host_data = None

        return guest_data, host_data    



