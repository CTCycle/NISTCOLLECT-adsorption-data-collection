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

    pass
    
  


    # # save data either locally or in a S3 bucket as .csv files
    # SC_dataset_expanded, BM_dataset_expanded = dataworker.dataset_expansion(SC_dataset, BM_dataset) 
    # file_loc = os.path.join(DATA_EXP_PATH, 'single_component_dataset.csv') 
    # SC_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index=False, sep=';', encoding='utf-8')
    # file_loc = os.path.join(DATA_EXP_PATH, 'binary_mixture_dataset.csv') 
    # BM_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index=False, sep=';', encoding='utf-8')

    # logger.info('NISTCOLLECT data collection has terminated. All files have been saved.')
