import numpy as np
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

        self.exp_features = ['temperature']
        self.guest_features = ['molecular_weight', 'elements', 'heavy_atoms', 'molecular_formula',
                               'SMILE', 'H_acceptors', 'H_donors']
        self.host_features = ['adsorbent_name']
        self.pressure_series = 'pressure_in_Pascal'
        self.uptake_series = 'uptake_in_mol_g'  


    #--------------------------------------------------------------------------
    def dataset_downsampling(self, dataset: pd.DataFrame):
       
        if CONFIG["dataset"]["SAMPLE_SIZE"] is not None: 
            sample_size = int(np.ceil(dataset.shape[0] * CONFIG["dataset"]["SAMPLE_SIZE"]))      
            dataset = dataset.sample(n=sample_size, random_state=CONFIG["dataset"]["SPLIT_SEED"])
            
        return dataset         
        
    #--------------------------------------------------------------------------
    def split_train_and_validation(self, dataset: pd.DataFrame):

        dataset = self.dataset_downsampling(dataset)
        inputs = dataset[[x for x in dataset.columns if x != self.uptake_series]]
        labels = dataset[self.uptake_series]
        train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=self.validation_size, 
                                                            random_state=CONFIG["SEED"], shuffle=True, 
                                                            stratify=None) 
        
        return train_X, test_X, train_Y, test_Y
    
    #--------------------------------------------------------------------------
    def isolate_inputs(self, X : pd.DataFrame):

        experimental_features = X[self.exp_features]
        guest_features = X[self.guest_features]
        host_features = X[self.host_features]
        pressures = X[self.pressure_series].values    
        
        return experimental_features, guest_features, host_features, pressures

   
