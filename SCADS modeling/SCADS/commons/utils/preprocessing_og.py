import numpy as np
import tensorflow as tf
from keras.api._v2.keras import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tqdm import tqdm
tqdm.pandas()
      

    

# [DATA PREPROCESSING]
#==============================================================================
# preprocess adsorption data
#==============================================================================
class PreProcessing:  


    def __init__(self):        
        self.parameters = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                           'H_acceptors', 'H_donors', 'heavy_atoms']
        self.ads_col, self.sorb_col  = ['adsorbent_name'], ['adsorbate_name'] 
        self.P_col, self.Q_col  = 'pressure_in_Pascal', 'uptake_in_mol_g'
        self.P_unit_col, self.Q_unit_col  = 'pressureUnits', 'adsorptionUnits'   

    #--------------------------------------------------------------------------
    def split_dataset(self, dataset, test_size, seed=42):
        inputs = dataset[[x for x in dataset.columns if x != self.Q_col]]
        labels = dataset[self.Q_col]
        train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=test_size, 
                                                            random_state=seed, shuffle=True, 
                                                            stratify=None) 
        
        return train_X, test_X, train_Y, test_Y       

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
    
    # encode variables  
    #--------------------------------------------------------------------------  
    def GH_encoding(self, unique_adsorbents, unique_sorbates, train_X, test_X):

        '''
        Encode categorical features using ordinal encoding. This method encodes categorical 
        features in the training and testing data using ordinal encoding.

        Keyword Arguments:
            unique_adsorbents (int): Number of unique adsorbents.
            unique_sorbates (int): Number of unique sorbates.
            train_X (DataFrame): DataFrame containing the features of the training data.
            test_X (DataFrame): DataFrame containing the features of the testing data.

        Returns:
            Tuple: A tuple containing the encoded training features and encoded testing features.
        
        '''      
        self.host_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=unique_adsorbents - 1)
        self.guest_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value',  unknown_value=unique_sorbates - 1)
        
        train_X[['adsorbent_name']] = self.host_encoder.fit_transform(train_X[['adsorbent_name']])
        train_X[['adsorbate_name']] = self.guest_encoder.fit_transform(train_X[['adsorbate_name']])
        test_X[['adsorbent_name']] = self.host_encoder.transform(test_X[['adsorbent_name']])
        test_X[['adsorbate_name']] = self.guest_encoder.transform(test_X[['adsorbate_name']])

        return train_X, test_X    
    
    #--------------------------------------------------------------------------  
    def sequence_padding(self, dataset, column, pad_value=-1, pad_length=50):
            
        dataset[column] = process.sequence.pad_sequences(dataset[column], 
                                                               maxlen=pad_length, 
                                                               value=pad_value, 
                                                               dtype='float32', 
                                                               padding='post').tolist()           

        return dataset 

    #--------------------------------------------------------------------------
    def create_tf_dataset(self, X_train, X_test, Y_train, Y_test, 
                          pad_length, batch_size=32, buffer_size=tf.data.AUTOTUNE):

        '''
        Creates a TensorFlow dataset from a generator. This function initializes 
        a TensorFlow dataset using a provided generator that yields batches of 
        inputs and targets. It sets up the dataset with an appropriate output 
        signature based on the first batch of data from the generator and applies 
        prefetching to improve data loading efficiency.

        Keyword Arguments:
            generator (Generator): A generator function or an instance with a `__getitem__` method that yields batches of data.
            buffer_size (int, optional): The number of elements to prefetch in the dataset. Default is `tf.data.AUTOTUNE`, allowing TensorFlow to automatically tune the buffer size.

        Returns:
            tf.data.Dataset: A TensorFlow dataset ready for model training or evaluation.

        '''
        # TRAIN DATASET
        parameters = X_train[self.parameters].values
        guests = X_train[self.sorb_col].values
        hosts = X_train[self.ads_col].values
        pressures = X_train[self.P_col].values
        uptakes = Y_train[self.Q_col].values
        # create datasets from tensor slices
        tf_parameters = tf.data.Dataset.from_tensor_slices(parameters)
        tf_guests = tf.data.Dataset.from_tensor_slices(guests)
        tf_hosts = tf.data.Dataset.from_tensor_slices(hosts)
        tf_pressures = tf.data.Dataset.from_tensor_slices([np.array(x).reshape(pad_length) for x in pressures])         
        tf_uptakes = tf.data.Dataset.from_tensor_slices([np.array(x).reshape(pad_length) for x in uptakes])  
        # create merged tf.dataset
        train_inputs = tf.data.Dataset.zip((tf_parameters, tf_hosts, tf_guests, tf_pressures))        
        train_dataset = tf.data.Dataset.zip((train_inputs, tf_uptakes))

        # TEST DATASET
        parameters = X_test[self.parameters].values
        guests = X_test[self.sorb_col].values
        hosts = X_test[self.ads_col].values
        pressures = X_test[self.P_col].values
        uptakes = Y_test[self.Q_col].values
        # create datasets from tensor slices
        tf_parameters = tf.data.Dataset.from_tensor_slices(parameters)
        tf_guests = tf.data.Dataset.from_tensor_slices(guests)
        tf_hosts = tf.data.Dataset.from_tensor_slices(hosts)
        tf_pressures = tf.data.Dataset.from_tensor_slices([np.array(x).reshape(pad_length) for x in pressures])        
        tf_uptakes = tf.data.Dataset.from_tensor_slices([np.array(x).reshape(pad_length) for x in uptakes])  
        # create merged tf.dataset
        test_inputs = tf.data.Dataset.zip((tf_parameters, tf_hosts, tf_guests, tf_pressures))        
        test_dataset = tf.data.Dataset.zip((test_inputs, tf_uptakes))

        train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=buffer_size) 
        test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=buffer_size)

        return train_dataset, test_dataset
        
    
    
          
