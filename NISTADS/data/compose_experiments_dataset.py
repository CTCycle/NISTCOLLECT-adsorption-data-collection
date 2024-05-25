import os
import sys
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [DEFINE PROJECT FOLDER PATH]
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir) 

# [IMPORT CUSTOM MODULES]
from utils.datamaker.datasets import AdsorptionDataset
from utils.NISTDB.core import NISTAdsorptionAPI
from config.pathfinder import DATA_EXP_PATH
import config.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [GET ISOTHERM EXPERIMENTS INDEX]
    #--------------------------------------------------------------------------
    print('\nCollect experiments data\n')

    # get isotherm indexes invoking API    
    webworker = NISTAdsorptionAPI()
    isotherm_index = webworker.Get_Isotherms_Index()
    isotherm_index = isotherm_index[:int(len(isotherm_index) * cnf.EXP_FRACTION)] 
    isotherm_names = [x['filename'] for x in isotherm_index]

    # create a dataframe with extracted experiments data    
    df_experiments = pd.DataFrame(isotherm_index)    
    print(f'Total number of adsorption experiments: {df_experiments.shape[0]}')
    print()

    # define a function to split the index into chunks to be processed sequentially,
    # to avoid excessive burden on system memory (for low RAM machines)    
    window_size = int(cnf.CHUNK_SIZE * len(isotherm_names))
    def list_fragmenter(lst, n):    
        for i in range(0, len(lst), n):
            yield lst[i:i + n]    

    # 2. [COLLECT DATA]
    #--------------------------------------------------------------------------
    drop_columns = ['category', 'tabular_data', 'isotherm_type', 'digitizer', 'articleSource']
    for i, fg in enumerate(list_fragmenter(isotherm_names, window_size)):
        isotherm_data = webworker.Get_Isotherms_Data(fg)
        df_isotherms = pd.DataFrame(isotherm_data).drop(columns = drop_columns)     
        dataworker = AdsorptionDataset(df_isotherms)

        # split the chunk dataset based on whether experiments is performed on single 
        # components or binary mixtures
        single_compound, binary_mixture = dataworker.split_by_mixcomplexity()

        # extracts the adsorption experiments data and expand the dataset
        SC_dataset = dataworker.extract_adsorption_data(single_compound, num_species=1) 
        BM_dataset = dataworker.extract_adsorption_data(binary_mixture, num_species=2)

        # save data either locally or in a S3 bucket as .csv files
        SC_dataset_expanded, BM_dataset_expanded = dataworker.dataset_expansion(SC_dataset, BM_dataset) 
        file_loc = os.path.join(DATA_EXP_PATH, 'single_component_dataset.csv') 
        SC_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index=False, sep=';', encoding='utf-8')
        file_loc = os.path.join(DATA_EXP_PATH, 'binary_mixture_dataset.csv') 
        BM_dataset_expanded.to_csv(file_loc, mode='a' if i>0 else 'w', index=False, sep=';', encoding='utf-8')
    
    print('NISTADS data collection has terminated. All files have been saved.')
