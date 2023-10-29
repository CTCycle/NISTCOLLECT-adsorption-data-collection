from tqdm import tqdm
import requests as r


# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class NISTAdsorptionAPI:  
    
    def __init__(self):        
        self.url_isotherms = 'https://adsorption.nist.gov/isodb/api/isotherms.json'
        self.url_adsorbates = 'https://adsorption.nist.gov/isodb/api/gases.json'
        self.url_adsorbents = 'https://adsorption.nist.gov/matdb/api/materials.json'
        
        
    # function to retrieve HTML data
    #==========================================================================
    def Get_GuestHost_Index(self):
        
        '''
        Extracts table with data from HTML
        
        Keyword arguments:        
            HTML_obj (BeautifulSoup): A webdriver instance.
                
        Returns:            
            main_data (BeautifulSoup): BeautifulSoup object containing the HTML content of the page
            
        ''' 
        adsorbents_json = r.get(self.url_adsorbents)
        adsorbates_json = r.get(self.url_adsorbates)        
        if adsorbents_json.status_code == 200:    
            adsorbents_index = adsorbents_json.json()       
        else:
            print(f'Error: Failed to retrieve adsorbents data. Status code: {adsorbents_json.status_code}')
            adsorbents_index = None        
        if adsorbates_json.status_code == 200:    
            adsorbates_index = adsorbates_json.json()       
        else:
            print(f'Error: Failed to retrieve adsorbates data. Status code: {adsorbates_json.status_code}')
            adsorbates_index = None         
            
        return adsorbates_index, adsorbents_index    
    
    
    # function to retrieve HTML data
    #==========================================================================
    def Get_GuestHost_Data(self, item_names, focus = 'guest'):

        '''
        Extracts the data of the guest or host from the HTML object.
    
        Keyword arguments:
            focus (str): A string indicating whether to extract data for the guest or host. Default is 'guest'.
    
        Returns:
            list: A list of dictionaries containing the data of the guest or host.
    
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
    
    
    # function to retrieve HTML data
    #==========================================================================
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
    #==========================================================================
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
                single_experiment = response.json()
                isotherms_data.append(single_experiment)
            except:
                self.notworking_urls.append(self.url_isotherms_byname)                
                
        return isotherms_data
    
    
            
        
            
       
            
                
                
