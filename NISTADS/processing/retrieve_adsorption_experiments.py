# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.dataloader.serializer import save_adsorption_datasets
from NISTADS.commons.utils.datafetch.experiments import AdsorptionDataFetch
from NISTADS.commons.utils.datamaker.datasets import BuildAdsorptionDataset
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':      

    # 1. [GET ISOTHERM EXPERIMENTS INDEX]
    #--------------------------------------------------------------------------
    # get isotherm indexes invoking API
    logger.info('Collect adsorption isotherm indexes')
    webworker = AdsorptionDataFetch()
    experiments_index = webworker.get_experiments_index()     

    # 2. [COLLECT ADSORPTION EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorption isotherms data')
    adsorption_data = webworker.get_experiments_data(experiments_index) 
        
    # 6. [PREPARE COLLECTED EXPERIMENTS DATA]
    #--------------------------------------------------------------------------    
    builder = BuildAdsorptionDataset()
    # remove excluded columns from the dataframe
    adsorption_data = builder.drop_excluded_columns(adsorption_data)
    # split current dataframe by complexity of the mixture (single component or binary mixture)
    single_component, binary_mixture = builder.split_by_mixture_complexity(adsorption_data) 
    # extract nested data in dataframe rows and reorganise them into columns
    single_component = builder.extract_nested_data(single_component)
    binary_mixture = builder.extract_nested_data(binary_mixture)
    # finally expand the dataset to represent each measurement with a single row
    # save the final version of the adsorption dataset
    single_component, binary_mixture = builder.expand_dataset(single_component, binary_mixture)
    save_adsorption_datasets(single_component, binary_mixture) 
    
    logger.info(f'Data collection is concluded, files have been saved in {DATA_PATH}')
  

    