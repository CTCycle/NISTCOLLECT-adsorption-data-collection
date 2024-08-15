# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.datascraper.materials import GuestHostAPI
from NISTADS.commons.utils.datamaker.properties import MolecularProperties
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [COLLECT GUEST/HOST INDEXES]
    #--------------------------------------------------------------------------
    # get guest and host indexes invoking API
    logger.info('Collect guest/host indexes')
    webworker = GuestHostAPI()
    guest_index, host_index = webworker.get_guest_host_index()     

    # 2. [COLLECT GUEST/HOST DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data')
    guest_data, host_data = webworker.get_guest_host_data(guest_index, host_index) 

    # 3. [ADD MOLECULAR PROPERTIES]
    #--------------------------------------------------------------------------
    enricher = MolecularProperties()
    guest_properties = enricher.get_properties_from_multiple_names(guest_data)
    enricher.save_properties_dataframe(guest_properties)
  

    
    
    