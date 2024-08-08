# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTCOLLECT.commons.utils.datascraper.experiments import AdsorptionDataAPI
from NISTCOLLECT.commons.utils.datamaker.datasets import DataProcessing
from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [GET ISOTHERM EXPERIMENTS INDEX]
    #--------------------------------------------------------------------------
    # get isotherm indexes invoking API
    logger.info('Collect adsorption isotherm indexes')
    webworker = AdsorptionDataAPI()
    experiments_index = webworker.get_experiments_index()     

    # 2. [COLLECT ADSORPTION EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data')
    adsorption_data = webworker.get_experiments_data(experiments_index) 
    
    # # 2. [COLLECT DATA]
    #--------------------------------------------------------------------------
    processor = DataProcessing()
    dataset = processor.process_dataset(adsorption_data)

    