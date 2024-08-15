import os
import numpy as np
import pandas as pd
from keras.api.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger



# [MERGE DATASETS]
###############################################################################
class SequenceProcessing:

    def __init__(self):

        self.P_TARGET_COL = 'pressure_in_Pascal' 
        self.Q_TARGET_COL = 'uptake_in_mol_g'

    #--------------------------------------------------------------------------
    def remove_leading_zeros(self, dataframe: pd.DataFrame):
        
        def _inner_function(row):
            pressure_series = row[self.P_TARGET_COL]
            uptake_series = row[self.Q_TARGET_COL]
            # Find the index of the first non-zero element or get the last index if all are zeros
            no_zero_index = next((i for i, x in enumerate(pressure_series) if x != 0), len(pressure_series) - 1)                
            # Determine how many leading zeros were removed
            zeros_removed = max(0, no_zero_index - 1)                
            processed_pressure_series = pressure_series[zeros_removed:]             
            processed_uptake_series = uptake_series[zeros_removed:]

            return pd.Series([processed_pressure_series, processed_uptake_series])

        dataframe[[self.P_TARGET_COL, self.Q_TARGET_COL]] = dataframe.apply(_inner_function, axis=1)
        
        return dataframe  


    #--------------------------------------------------------------------------  
    def sequence_padding(self, dataset, column, pad_value=-1, pad_length=50):
            
        dataset[column] = sequence.pad_sequences(dataset[column], 
                                                maxlen=pad_length, 
                                                value=pad_value, 
                                                dtype='float32', 
                                                padding='post').tolist()           

        return dataset

       
    
    # normalize sequences using a RobustScaler: X = X - median(X)/IQR(X)
    # flatten and reshape array to make it compatible with the scaler
    #--------------------------------------------------------------------------  
    def normalize_sequences(self, train, test, column):        
        
        normalizer = MinMaxScaler(feature_range=(0,1))
        sequence_array = np.array([item for sublist in train[column] for item in sublist]).reshape(-1, 1)         
        normalizer.fit(sequence_array)
        train[column] = train[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())
        test[column] = test[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())

        return train, test, normalizer
    
    # normalize parameters
    #--------------------------------------------------------------------------  
    def normalize_parameters(self, train_X, train_Y, test_X, test_Y):

        '''
        Normalize the input features and output labels for training and testing data.
        This method normalizes the input features and output labels to facilitate 
        better model training and evaluation.

        Keyword Arguments:
            train_X (DataFrame): DataFrame containing the features of the training data.
            train_Y (list): List containing the labels of the training data.
            test_X (DataFrame): DataFrame containing the features of the testing data.
            test_Y (list): List containing the labels of the testing data.

        Returns:
            Tuple: A tuple containing the normalized training features, normalized training labels,
                   normalized testing features, and normalized testing labels.
        
        '''        
        # cast float type for both the labels and the continuous features columns 
        norm_columns = ['temperature', 'mol_weight', 'complexity', 'heavy_atoms']       
        train_X[norm_columns] = train_X[norm_columns].astype(float)        
        test_X[norm_columns] = test_X[norm_columns].astype(float)
        
        # normalize the numerical features (temperature and physicochemical properties)      
        self.param_normalizer = MinMaxScaler(feature_range=(0, 1))
        train_X[norm_columns] = self.param_normalizer.fit_transform(train_X[norm_columns])
        test_X[norm_columns] = self.param_normalizer.transform(test_X[norm_columns])        

        return train_X, train_Y, test_X, test_Y 

