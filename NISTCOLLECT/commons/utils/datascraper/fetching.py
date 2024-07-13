import pandas as pd
from tqdm import tqdm
import requests as r

from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# [NIST DATABASE API]
#------------------------------------------------------------------------------
class GuestHostAPI: 

    def __init__(self):      
        self.url_GUEST = 'https://adsorption.nist.gov/isodb/api/gases.json'
        self.url_HOST = 'https://adsorption.nist.gov/matdb/api/materials.json'
        self.guest_fraction = CONFIG["GUEST_FRACTION"]
        self.host_fraction = CONFIG["HOST_FRACTION"]
        self.guest_identifier = 'InChIKey'
        self.host_identifier = 'hashkey'


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
            df_guest = pd.DataFrame()
        if host_json.status_code == 200:
            host_index = host_json.json() 
            df_host = pd.DataFrame(host_index) 
            logger.info(f'Total number of adsorbates: {df_host.shape[0]}')
        else:
            logger.error(f'Failed to retrieve adsorbates data. Status code: {host_json.status_code}')
            df_host = pd.DataFrame() 
  
        return df_guest, df_host 


    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_guest_host_data(self, df_guest, df_host):

        guest_names = df_guest[self.guest_identifier].to_list()
        host_names = df_host[self.host_identifier].to_list()

        # fetch guest data using the names in the dataframe as reference        
        guest_data = []
        for name in tqdm(guest_names):            
            named_guest_url = f'https://adsorption.nist.gov/isodb/api/gas/{name}.json'
            guest_response = r.get(named_guest_url)
            if guest_response.status_code != 200:
                logger.error(f'Could not fetch data from {named_guest_url}')
                continue

            guest_data.append(guest_response.json())      

    
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_materials_data(self, item_names, focus='guest'):

        '''
        Retrieves the data of the specified guests or hosts from the NIST ISODB.
        This function iterates over the provided item names and sends GET requests 
        to the NIST ISODB API to retrieve the data for each item. 
        The focus parameter determines whether the function retrieves data for guests or hosts. 
        If the request is successful, the JSON response is converted to a Python 
        dictionary and added to the list of extracted data. If the request fails, 
        an error message is printed and the function continues to the next item.

        Keyword arguments:
            item_names (list): A list of strings representing the names of the items to retrieve data for.
            focus (str): A string indicating whether to extract data for the 'guest' or 'host'. Default is 'guest'.

        Returns:
            list: A list of dictionaries where each dictionary contains the data for a single guest or host. 
                  If a request fails, the corresponding item in the list will be None.

        '''          
        extracted_data = []        
        if focus == 'guest':            
            for name in tqdm(item_names):            
                self.url_guest_byname = f'https://adsorption.nist.gov/isodb/api/gas/{name}.json'
                try:
                    response = r.get(self.url_guest_byname)
                    guest_entry = response.json()
                    extracted_data.append(guest_entry)
                except:
                    print('Failed to retrieve data')
                    guest_entry = None       
        
        elif focus == 'host':
            for name in tqdm(item_names):                        
                self.url_host_byname = f'https://adsorption.nist.gov/matdb/api/material/{name}.json'
                try:
                    response = r.get(self.url_host_byname)
                    host_entry = response.json()
                    extracted_data.append(host_entry)
                except:
                    print('Failed to retrieve data')
                    host_entry = None        
       
        return extracted_data   

# [NIST DATABASE API]
#------------------------------------------------------------------------------
class NISTAdsorptionAPI:  
    
    def __init__(self):        
        self.url_isotherms = 'https://adsorption.nist.gov/isodb/api/isotherms.json'
        



            
        
    
    
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def Get_Isotherms_Index(self):
        
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
        else:
            print(f'Error: Failed to retrieve data. Status code: {response.status_code}')
            isotherm_index = None
            
        return isotherm_index   
    
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def Get_Isotherms_Data(self, exp_names): 

        '''        
        This function retrieves isotherm data for a given list of experiment names 
        from the NIST ISODB, sending a GET request to the database for each experiment 
        name and appending the returned JSON data to a list. If the request fails 
        for any experiment name, its URL is added to a list of non-working URLs.

        Keyword arguments:
            exp_names (list): A list of experiment names for which isotherm data is to be retrieved.

        Returns:
            list: A list of dictionaries where each dictionary contains the isotherm data for an experiment.            
        
        '''  
        self.notworking_urls = []     
        isotherms_data = []
        for name in tqdm(exp_names):
            try:
                self.url_isotherms_byname = f'https://adsorption.nist.gov/isodb/api/isotherm/{name}.json'
                response = r.get(self.url_isotherms_byname)                
                isotherms_data.append(response.json())
            except:
                self.notworking_urls.append(self.url_isotherms_byname)                
                
        return isotherms_data
    
    
            
        
            
       
            
                
                
