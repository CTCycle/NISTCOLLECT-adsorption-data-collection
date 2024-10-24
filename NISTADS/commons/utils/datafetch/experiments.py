import numpy as np
import pandas as pd
import requests as r
import asyncio

from NISTADS.commons.utils.datafetch.status import GetServerStatus
from NISTADS.commons.utils.datafetch.asynchronous import data_from_multiple_URLs
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


# [CHECK SERVER STATUS]
###############################################################################
server = GetServerStatus()
server.check_status()


# [NIST DATABASE API]
###############################################################################
class AdsorptionDataFetch:  
    
    def __init__(self):        
        self.exp_fraction = CONFIG["collection"]["EXP_FRACTION"]
        self.max_parallel_calls = CONFIG["collection"]["PARALLEL_TASKS_EXP"]
        self.url_isotherms = 'https://adsorption.nist.gov/isodb/api/isotherms.json'
        self.exp_identifier = 'filename'
    
    
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
            df_isotherms = pd.DataFrame(isotherm_index) 
            logger.info(f'Successfully retrieved adsorption isotherm index from {self.url_isotherms}')    
        else:
            logger.error(f'Error: Failed to retrieve data. Status code: {response.status_code}')
            df_isotherms = None
            
        return df_isotherms
    
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_experiments_data(self, df_isotherms : pd.DataFrame): 

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
        exp_samples = int(np.ceil(self.exp_fraction * df_isotherms.shape[0]))

        df_experiments = None
        if df_isotherms is not None:
            loop = asyncio.get_event_loop()
            exp_names = df_isotherms[self.exp_identifier].to_list()[:exp_samples]
            exp_URLs = [f'https://adsorption.nist.gov/isodb/api/isotherm/{name}.json' for name in exp_names]
            exp_data = loop.run_until_complete(data_from_multiple_URLs(exp_URLs, self.max_parallel_calls))
            exp_data = [data for data in exp_data if data is not None]
            df_experiments = pd.DataFrame(exp_data)        

        return df_experiments

             