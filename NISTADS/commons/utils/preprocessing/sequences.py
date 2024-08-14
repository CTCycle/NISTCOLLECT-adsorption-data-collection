import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger



# [MERGE DATASETS]
###############################################################################
def remove_leading_zeros(self, sequence_A, sequence_B):

    # Find the index of the first non-zero element or get the last index if all are zeros
    first_non_zero_index_A = next((i for i, x in enumerate(sequence_A) if x != 0), len(sequence_A) - 1)
    first_non_zero_index_B = next((i for i, x in enumerate(sequence_B) if x != 0), len(sequence_B) - 1)
            
    # Ensure to remove leading zeros except one, for both sequences
    processed_seq_A = sequence_A[max(0, first_non_zero_index_A - 1):]
    processed_seq_B = sequence_B[max(0, first_non_zero_index_B - 1):]        
        
    return processed_seq_A, processed_seq_B   

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

