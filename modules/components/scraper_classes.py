from tqdm import tqdm
from selenium import webdriver
import requests as r


# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class WebDriver:
    
    '''
    Initializes a webdriver instance with Chrome options set to disable images loading.
    
    Keyword arguments:    
        wd_path (str): The file path to the Chrome webdriver executable  
    
    '''
    def __init__(self, wd_path):
        self.wd_path = wd_path
        self.option = webdriver.ChromeOptions()
        self.chrome_prefs = {}
        self.option.experimental_options['prefs'] = self.chrome_prefs
        self.chrome_prefs['profile.default_content_settings'] = {'images': 2}
        self.chrome_prefs['profile.managed_default_content_settings'] = {'images': 2}  
        
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
            self.adsorbents_index = adsorbents_json.json()       
        else:
            print(f'Error: Failed to retrieve adsorbents data. Status code: {adsorbents_json.status_code}')
            self.adsorbents_index = None
        
        if adsorbates_json.status_code == 200:    
            self.adsorbates_index = adsorbates_json.json()       
        else:
            print(f'Error: Failed to retrieve adsorbates data. Status code: {adsorbates_json.status_code}')
            self.adsorbates_index = None         
            
        return self.adsorbates_index, self.adsorbents_index
    
    # function to retrieve HTML data
    #==========================================================================
    def Get_GuestHost_Names(self):

        '''
        Extracts the names of the guest and host from the HTML object.
    
        Returns:
            tuple: A tuple containing two lists of strings. The first list contains 
                   the names of the guests and the second list contains the names of the hosts.
        
        '''        
        self.guest_names = []
        self.host_names = []
        for data in self.adsorbates_index:
            name = data['InChIKey']           
            self.guest_names.append(name)
            
        for data in self.adsorbents_index:
            name = data['hashkey']           
            self.host_names.append(name)
            
        return self.guest_names, self.host_names
    
    # function to retrieve HTML data
    #==========================================================================
    def Get_GuestHost_Data(self, focus = 'guest'):

        '''
        Extracts the data of the guest or host from the HTML object.
    
        Keyword arguments:
            focus (str): A string indicating whether to extract data for the guest or host. Default is 'guest'.
    
        Returns:
            list: A list of dictionaries containing the data of the guest or host.
    
        '''          
        self.extracted_data = []        
        if focus == 'guest':            
            for name in tqdm(self.guest_names):            
                self.url_guest_byname = f'https://adsorption.nist.gov/isodb/api/gas/{name}.json'
                try:
                    response = r.get(self.url_guest_byname)
                    guest_entry = response.json()
                    self.extracted_data.append(guest_entry)
                except:
                    print('Failed to retrieve data')
                    guest_entry = None       
        
        elif focus == 'host':
            for name in tqdm(self.host_names):                        
                self.url_host_byname = f'https://adsorption.nist.gov/matdb/api/material/{name}.json'
                try:
                    response = r.get(self.url_host_byname)
                    host_entry = response.json()
                    self.extracted_data.append(host_entry)
                except:
                    print('Failed to retrieve data')
                    host_entry = None        
       
        return self.extracted_data
    
    
    
    # function to retrieve HTML data
    #==========================================================================
    def Get_Isotherms_Index(self):
        
        '''
        Extracts table with data from HTML
        
        Keyword arguments:        
            HTML_obj (BeautifulSoup): A webdriver instance.
                
        Returns:            
            main_data (BeautifulSoup): BeautifulSoup object containing the HTML content of the page
            
        ''' 
        response = r.get(self.url_isotherms)
        if response.status_code == 200:    
            self.isotherm_index = response.json()       
        else:
            print(f'Error: Failed to retrieve data. Status code: {response.status_code}')
            self.isotherm_index = None
            
        return self.isotherm_index
    
    # function to retrieve HTML data
    #==========================================================================
    def Get_Isotherms_Names(self):

        '''
        Extracts the names of the isotherms from the HTML object.
    
        Returns:
            list: A list of strings containing the names of the isotherms.
        
        '''        
        self.exp_names = []
        for data in self.isotherm_index:
            
            exp_name = data['filename']
            self.exp_names.append(exp_name)
            
        return self.exp_names
    
    # function to retrieve HTML data
    #==========================================================================
    def Get_Isotherms_Data(self): 

        '''
        Extracts the data of the isotherms from the HTML object.
    
        Returns:
            list: A list of dictionaries containing the data of the isotherms.
    
        '''  
        self.notworking_urls = []     
        self.isotherms_data = []
        for name in tqdm(self.exp_names):
            try:
                self.url_isotherms_byname = f'https://adsorption.nist.gov/isodb/api/isotherm/{name}.json'
                response = r.get(self.url_isotherms_byname)
                single_experiment = response.json()
                self.isotherms_data.append(single_experiment)
            except:
                self.notworking_urls.append(self.url_isotherms_byname)                
                
        return self.isotherms_data
    
    
            
        
            
       
            
                
                
