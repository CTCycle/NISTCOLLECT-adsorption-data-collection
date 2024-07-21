import os
import sys
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTCOLLECT.commons.utils.datascraper.experiments import AdsorptionDataAPI
from NISTCOLLECT.commons.utils.datamaker.properties import MolecularProperties
from NISTCOLLECT.commons.constants import CONFIG, DATA_MAT_PATH
from NISTCOLLECT.commons.logger import logger


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [GET ISOTHERM EXPERIMENTS INDEX]
    #--------------------------------------------------------------------------
    # get isotherm indexes invoking API
    logger.info('Collect guest/host indexes')
    webworker = AdsorptionDataAPI()
    experiment_index = webworker.get_experiments_indexes()     

    # 2. [COLLECT ADSORPTION EXPERIMENTS DATA]
    #--------------------------------------------------------------------------
    logger.info('Extracting adsorbents and sorbates data')
    guest_data, host_data = webworker.get_experiments_data(experiment_index) 

    # # 3. [ADD MOLECULAR PROPERTIES]
    # #--------------------------------------------------------------------------
    # enricher = MolecularProperties()
    # df_properties = enricher.extract_molecular_properties(guest_data)

    # # define a function to split the index into chunks to be processed sequentially,
    # # to avoid excessive burden on system memory (for low RAM machines)    
    # window_size = int(cnf.CHUNK_SIZE * len(isotherm_names))
    # def list_fragmenter(lst, n):    
    #     for i in range(0, len(lst), n):
    #         yield lst[i:i + n]    

    # # 2. [COLLECT DATA]
    # #--------------------------------------------------------------------------
    # drop_columns = ['category', 'tabular_data', 'isotherm_type', 'digitizer', 'articleSource']
    # for i, fg in enumerate(list_fragmenter(isotherm_names, window_size)):
    #     isotherm_data = webworker.Get_Isotherms_Data(fg)
    #     df_isotherms = pd.DataFrame(isotherm_data).drop(columns = drop_columns)     
    #     dataworker = AdsorptionDataset(df_isotherms)

    #     # split the chunk dataset based on whether experiments is performed on single 
    #     # components or binary mixtures
    #     single_compound, binary_mixture = dataworker.split_by_mixcomplexity()

    #     # extracts the adsorption experiments data and expand the dataset
    #     SC_dataset = dataworker.extract_adsorption_data(single_compound, num_species=1) 
    #     BM_dataset = dataworker.extract_adsorption_data(binary_mixture, num_species=2)

    #     # save data either locally or in a S3 bucket as .csv files
    #     SC_dataset_expanded, BM_dataset_expanded = dataworker.dataset_expansion(SC_dataset, BM_dataset) 
    #     file_loc = os.path.join(DATA_EXP_PATH, 'single_component_dataset.csv') 
    #     SC_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index=False, sep=';', encoding='utf-8')
    #     file_loc = os.path.join(DATA_EXP_PATH, 'binary_mixture_dataset.csv') 
    #     BM_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index=False, sep=';', encoding='utf-8')
    
    # print('NISTADS data collection has terminated. All files have been saved.')
