# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.datascraper.experiments import AdsorptionDataAPI
from NISTADS.commons.utils.datamaker.datasets import AdsorptionDatasetPreparation
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


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
    logger.info('Extracting adsorption isotherms data')
    adsorption_data = webworker.get_experiments_data(experiments_index) 
    
    # # 2. [COLLECT DATA]
    #--------------------------------------------------------------------------
    processor = AdsorptionDatasetPreparation()
    dataset = processor.prepare_dataset(adsorption_data)

    