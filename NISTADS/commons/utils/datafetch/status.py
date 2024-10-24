import sys
import time
import requests as r

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# function to retrieve HTML data
###############################################################################
class GetServerStatus:

    def __init__(self):
        self.server_url = 'https://adsorption.nist.gov'

    #--------------------------------------------------------------------------
    def check_status(self):
       
        response = r.get(self.server_url)
        # Checking if the request was successful
        if response.status_code == 200:
            logger.info(f'NIST server is up and running. Status code: {response.status_code}')
        else:            
            logger.error(f'Failed to reach the server. Status code: {response.status_code}') 
            time.sleep(5)
            sys.exit()

    
        
            
       
            
                
                
