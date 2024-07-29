import os
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import asyncio

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTCOLLECT.commons.utils.datascraper.materials import GuestHostAPI
from NISTCOLLECT.commons.utils.datamaker.properties import MolecularProperties
from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [COLLECT GUEST/HOST INDEXES]
    #--------------------------------------------------------------------------
    # get isotherm indexes invoking API
    logger.info('Collect guest/host indexes')
    webworker = GuestHostAPI()
    guest_index, host_index = webworker.get_guest_host_indexes()     

    # 2. [COLLECT GUEST/HOST DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data')
    guest_data, host_data = webworker.get_guest_host_data(guest_index, host_index) 

    # 3. [ADD MOLECULAR PROPERTIES]
    #--------------------------------------------------------------------------
    enricher = MolecularProperties()
    properties = enricher.get_properties_from_multiple_names(guest_data)

    
      
    
    