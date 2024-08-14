import pandas as pd
from sklearn.model_selection import train_test_split

from XREPORT.commons.constants import CONFIG
from XREPORT.commons.logger import logger

# [DATA SPLITTING]
###############################################################################
class DatasetSplit:

    def __init__(self, ):

        # Set the sizes for the train and validation datasets       
        self.sample_size = CONFIG["dataset"]["SAMPLE_SIZE"]         
        self.validation_size = CONFIG["dataset"]["VALIDATION_SIZE"]
        self.train_size = 1.0 - self.validation_size

        self.Q_TARGET_COL = 'uptake_in_mol_g'

    #--------------------------------------------------------------------------
    def dataset_downsampling(self, dataset: pd.DataFrame):
       
        if CONFIG["dataset"]["SAMPLE_SIZE"] is not None:             
            dataset = dataset.sample(n=CONFIG["dataset"]["SAMPLE_SIZE"], 
                                     random_state=CONFIG["dataset"]["SPLIT_SEED"])
            
        return dataset       
        
        
    #--------------------------------------------------------------------------
    def split_dataset(self, dataset: pd.DataFrame):


        dataset = self.dataset_downsampling(dataset)
        inputs = dataset[[x for x in dataset.columns if x != self.Q_TARGET_COL]]
        labels = dataset[self.Q_TARGET_COL]
        train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=self.validation_size, 
                                                            random_state=CONFIG["SEED"], shuffle=True, 
                                                            stratify=None) 
        
        return train_X, test_X, train_Y, test_Y

   
