# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import save_materials_datasets
from NISTADS.commons.utils.datamaker.datasets import GetSpeciesFromExperiments
from NISTADS.commons.utils.datafetch.materials import GuestHostDataFetch

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':    
   

    # 1. [COLLECT GUEST/HOST INDEXES]
    #--------------------------------------------------------------------------
    # get guest and host indexes invoking API
    logger.info('Collect guest and host indexes from NIST DB')
    webworker = GuestHostDataFetch()
    guest_index, host_index = webworker.get_guest_host_index()     

    # 2. [COLLECT GUEST/HOST DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data from relative indexes')
    guest_data, host_data = webworker.get_guest_host_data(guest_index, host_index)

    # 3. [CHECK FOR EXPERIMENTS DATASET AND SYNC]
    #--------------------------------------------------------------------------
    extractor = GetSpeciesFromExperiments()
    species_names = extractor.extract_species_names()
    
    
    # 3. [PROCESS MATERIALS DATA]
    #--------------------------------------------------------------------------
    


    save_materials_datasets(guest_data, host_data)

    
   

    