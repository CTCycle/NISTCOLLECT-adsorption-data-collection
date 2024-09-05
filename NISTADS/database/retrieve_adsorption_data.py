# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.datascraper.materials import GuestHostAPI
from NISTADS.commons.utils.datascraper.experiments import AdsorptionDataAPI
from NISTADS.commons.utils.datamaker.properties import FetchMolecularProperties
from NISTADS.commons.utils.datamaker.datasets import AdsorptionDatasetPreparation
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
    webworker.save_dataframe(guest_data, host_data)

    # 3. [GET ISOTHERM EXPERIMENTS INDEX]
    #--------------------------------------------------------------------------
    # get isotherm indexes invoking API
    logger.info('Collect adsorption isotherm indexes')
    webworker = AdsorptionDataAPI()
    experiments_index = webworker.get_experiments_index()     

    # 4. [COLLECT ADSORPTION EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorption isotherms data')
    adsorption_data = webworker.get_experiments_data(experiments_index) 


    # 3. [ADD MOLECULAR PROPERTIES]
    #--------------------------------------------------------------------------
    property = FetchMolecularProperties()
    guest_properties = property.get_guest_properties(guest_data)



    
    # 5. [PREPARE COLLECTED EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    processor = AdsorptionDatasetPreparation()
    dataset = processor.prepare_dataset(adsorption_data)
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')
  

    