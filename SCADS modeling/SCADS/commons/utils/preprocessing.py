import os
import pandas as pd
import numpy as np
import pickle as pk
import tensorflow as tf
from keras.api._v2.keras import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tqdm import tqdm
tqdm.pandas()
      


def join_str(x):
    return ' '.join(x.astype(str))


# [GENERATIVE MODELS]
#==============================================================================
class PreProcessing:

    def __init__(self, max_P, max_Q):

        self.max_P = max_P
        self.max_Q = max_Q

        self.ADS_COL, self.SORB_COL  = ['adsorbent_name'], ['adsorbate_name'] 
        self.P_COL, self.Q_COL  = 'pressure_in_Pascal', 'uptake_in_mol_g'
        self.P_UNIT_COL, self.Q_UNIT_COL  = 'pressureUnits', 'adsorptionUnits' 
        self.VALID_UNITS = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 'cm3(STP)/g',
                            'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g']
                                        
        self.PARAMETERS = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                           'H_acceptors', 'H_donors', 'heavy_atoms']
        
        self.aggregate_dict = {'temperature' : 'first',                  
                                'adsorbent_name' : 'first',
                                'adsorbate_name' : 'first',                  
                                'complexity' : 'first',                  
                                'mol_weight' : 'first',
                                'covalent_units' : 'first',
                                'H_acceptors' : 'first',
                                'H_donors' : 'first',
                                'heavy_atoms' : 'first', 
                                'pressure_in_Pascal' : join_str,
                                'uptake_in_mol_g' : join_str}
        
    #--------------------------------------------------------------------------
    def split_dataset(self, dataset, test_size, seed=42):
        inputs = dataset[[x for x in dataset.columns if x != self.Q_COL]]
        labels = dataset[self.Q_COL]
        train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=test_size, 
                                                            random_state=seed, shuffle=True, 
                                                            stratify=None) 
        
        return train_X, test_X, train_Y, test_Y    

    #--------------------------------------------------------------------------
    def units_normalization(self, dataset):

        # filter experiments by pressure and uptake units 
        dataset = dataset[dataset[self.Q_UNIT_COL].isin(self.VALID_UNITS)]

        # convert pressures to Pascal
        dataset[self.P_COL] = dataset.progress_apply(lambda x : self.pressure_converter(x[self.P_UNIT_COL], 
                                                                                        x['pressure']), 
                                                                                        axis=1)
        # convert uptakes to mol/g
        dataset[self.Q_COL] = dataset.progress_apply(lambda x : self.uptake_converter(x[self.Q_UNIT_COL], 
                                                                                      x['adsorbed_amount'],
                                                                                      x['mol_weight']), axis=1)

        # further filter the dataset to remove experiments which values are outside desired boundaries, 
        # such as experiments with negative temperature, pressure and uptake values   
        dataset = dataset[dataset['temperature'].astype(int) > 0]
        dataset = dataset[dataset[self.P_COL].astype(float).between(0.0, self.max_P)]
        dataset = dataset[dataset[self.Q_COL].astype(float).between(0.0, self.max_Q)]

        return dataset
        

    #--------------------------------------------------------------------------
    def pressure_converter(self, type, p_val):

        '''
        Converts pressure from the specified unit to Pascals.

        Keyword arguments:
            type (str): The original unit of pressure.
            p_val (int or float): The original pressure value.

        Returns:
            p_val (int): The pressure value converted to Pascals.

        '''         
        if type == 'bar':
            p_val = int(p_val * 100000)        
                    
        return p_val

    #--------------------------------------------------------------------------
    def uptake_converter(self, q_unit, q_val, mol_weight):

        '''
        Converts the uptake value from the specified unit to moles per gram.

        Keyword arguments:
            q_unit (str):              The original unit of uptake.
            q_val (int or float):      The original uptake value.
            mol_weight (int or float): The molecular weight of the adsorbate.

        Returns:
            q_val (float): The uptake value converted to moles per gram

        '''        
        if q_unit in ('mmol/g', 'mol/kg'):
            q_val = q_val/1000         
        elif q_unit == 'mmol/kg':
            q_val = q_val/1000000
        elif q_unit == 'mg/g':
            q_val = q_val/1000/float(mol_weight)            
        elif q_unit == 'g/g':
            q_val = (q_val/float(mol_weight))                                   
        elif q_unit == 'wt%':                
            q_val = ((q_val/100)/float(mol_weight))          
        elif q_unit in ('g Adsorbate / 100g Adsorbent', 'g/100g'):              
            q_val = ((q_val/100)/float(mol_weight))                            
        elif q_unit in ('ml(STP)/g', 'cm3(STP)/g'):
            q_val = q_val/22.414      
                    
        return q_val 

    #--------------------------------------------------------------------------
    def add_guest_properties(self, df_isotherms, df_adsorbates):

        '''
        Assigns properties to adsorbates based on their isotherm data.

        This function takes two pandas DataFrames: one containing isotherm data (df_isotherms)
        and another containing adsorbate properties (df_adsorbates). It merges the two DataFrames
        on the 'adsorbate_name' column, assigns properties to each adsorbate, and returns a new
        DataFrame containing the merged data with assigned properties.

        Keyword Arguments:
            df_isotherms (pandas DataFrame): A DataFrame containing isotherm data.
            df_adsorbates (pandas DataFrame): A DataFrame containing adsorbate properties.

        Returns:
            df_adsorption (pandas DataFrame): A DataFrame containing merged isotherm data
                                                with assigned adsorbate properties.

        '''
        
        df_isotherms[self.ADS_COL] = df_isotherms[self.ADS_COL].apply(lambda x : str(x).lower()) 
        df_isotherms[self.SORB_COL] = df_isotherms[self.SORB_COL].apply(lambda x : str(x).lower())       
        df_properties = df_adsorbates[[x for x in self.PARAMETERS if x != 'temperature']]        
        df_adsorption = pd.merge(df_isotherms, df_properties, on=self.SORB_COL, how='inner')

        return df_adsorption

    #--------------------------------------------------------------------------
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
    def GH_encoding(self, train_X, test_X):

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
        
        self.host_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', 
                                           unknown_value=-1)
        self.guest_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value',  
                                            unknown_value=-1)
        
        train_X[[self.ADS_COL]] = self.host_encoder.fit_transform(train_X[[self.ADS_COL]])
        train_X[[self.SORB_COL]] = self.guest_encoder.fit_transform(train_X[[self.SORB_COL]])
        test_X[[self.ADS_COL]] = self.host_encoder.transform(test_X[[self.ADS_COL]])
        test_X[[self.SORB_COL]] = self.guest_encoder.transform(test_X[[self.SORB_COL]])

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
        parameters = X_train[self.PARAMETERS].values
        guests = X_train[self.SORB_COL].values
        hosts = X_train[self.ADS_COL].values
        pressures = X_train[self.P_COL].values
        uptakes = Y_train[self.Q_COL].values
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
        parameters = X_test[self.PARAMETERS].values
        guests = X_test[self.SORB_COL].values
        hosts = X_test[self.ADS_COL].values
        pressures = X_test[self.P_COL].values
        uptakes = Y_test[self.Q_COL].values
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
        
        
    
    
    


# [PREPROCESSING DATA PIPELINE]
#==============================================================================
class PreProcessPipeline:

    def __init__(self, max_P, max_Q, pad_value, pad_length, batch_size, 
                 test_size, split_seed, path):

        self.pad_value = pad_value
        self.pad_length = pad_length
        self.batch_size = batch_size
        self.test_size = test_size
        self.split_seed = split_seed
        self.path = path
        self.processor = PreProcessing(max_P, max_Q)       

    def save_processed_data(self, train_X, test_X, train_Y, test_Y,
                            param_normalizer, pressure_normalizer,
                            uptake_normalizer, host_encoder, guest_encoder):

        # save normalizers and encoders       
        normalizer_path = os.path.join(self.path, 'parameters_normalizer.pkl')
        with open(normalizer_path, 'wb') as file:
            pk.dump(param_normalizer, file)
        normalizer_path = os.path.join(self.path, 'pressure_normalizer.pkl')
        with open(normalizer_path, 'wb') as file:
            pk.dump(pressure_normalizer, file)
        normalizer_path = os.path.join(self.path, 'uptake_normalizer.pkl')
        with open(normalizer_path, 'wb') as file:
            pk.dump(uptake_normalizer, file)
        encoder_path = os.path.join(self.path, 'host_encoder.pkl')
        with open(encoder_path, 'wb') as file:
            pk.dump(host_encoder, file) 
        encoder_path = os.path.join(self.path, 'guest_encoder.pkl')
        with open(encoder_path, 'wb') as file:
            pk.dump(guest_encoder, file) 

        # save .csv files       
        # at first, convert sequences into single strings
        train_X[self.processor.P_COL] = train_X[self.processor.P_COL].apply(lambda x : ' '.join([str(f) for f in x])) 
        test_X[self.processor.P_COL] = test_X[self.processor.P_COL].apply(lambda x : ' '.join([str(f) for f in x]))
        train_Y[self.processor.Q_COL] = train_Y[self.processor.Q_COL].apply(lambda x : ' '.join(x)) 
        test_Y = test_Y.apply(lambda x : ' '.join(x)) 

        # save files
        filename = os.path.join(self.path, 'X_train.csv')
        train_X.to_csv(filename, sep=';', encoding='utf-8')
        filename = os.path.join(self.path, 'X_test.csv')
        test_X.to_csv(filename, sep=';', encoding='utf-8')
        filename = os.path.join(self.path, 'Y_train.csv')
        train_Y.to_csv(filename, sep=';', encoding='utf-8')
        filename = os.path.join(self.path, 'Y_test.csv')
        test_Y.to_csv(filename, sep=';', encoding='utf-8')
    
    def run_pipeline(self, df):  
        

        # transform series from unique string to lists    
        df[self.processor.P_COL] = df[self.processor.P_COL].apply(lambda x : [float(f) for f in x.split()])
        df[self.processor.Q_COL] = df[self.processor.Q_COL].apply(lambda x : [float(f) for f in x.split()])

        # split dataset in train and test subsets       
        train_X, test_X, train_Y, test_Y = self.processor.split_dataset(df, self.test_size, self.split_seed)
        self.num_train_samples = train_X.shape[0]
        self.num_test_samples = test_X.shape[0]

        # [PREPROCESS DATASET: NORMALIZING AND ENCODING]
        # determine number of unique adsorbents and adsorbates from the train dataset        
        print('\nEncoding categorical variables')
        unique_adsorbents = train_X[self.processor.ADS_COL].nunique() + 1
        unique_sorbates = train_X[self.processor.SORB_COL].nunique() + 1

        print(unique_adsorbents)

        # extract pretrained encoders to numerical indexes
        train_X, test_X = self.processor.GH_encoding(train_X, test_X)
        self.host_encoder = self.processor.host_encoder
        self.guest_encoder = self.processor.guest_encoder

        # normalize parameters (temperature, physicochemical properties) and sequences        
        print('\nNormalizing continuous variables (temperature, physicochemical properties)\n')
        train_X, train_Y, test_X, test_Y = self.processor.normalize_parameters(train_X, train_Y.to_frame(), 
                                                                               test_X, test_Y.to_frame())

        # normalize sequences of pressure and uptake
        train_X, test_X, pressure_normalizer = self.processor.normalize_sequences(train_X, test_X, self.processor.P_COL)
        train_Y, test_Y, uptake_normalizer = self.processor.normalize_sequences(train_Y, test_Y, self.processor.Q_COL)
        param_normalizer = self.processor.param_normalizer

        # apply padding to the pressure and uptake series (default value is -1 to avoid
        # interfering with real values)        
        train_X = self.processor.sequence_padding(train_X, self.processor.P_COL, self.pad_value, self.pad_length)
        test_X = self.processor.sequence_padding(test_X, self.processor.P_COL, self.pad_value, self.pad_length)
        train_Y = self.processor.sequence_padding(train_Y, self.processor.Q_COL, self.pad_value, self.pad_length)
        test_Y = self.processor.sequence_padding(test_Y, self.processor.Q_COL, self.pad_value, self.pad_length)

        # generate tf.datasets         
        train_dataset, test_dataset = self.processor.create_tf_dataset(train_X, test_X,
                                                                       train_Y, test_Y,
                                                                       self.pad_length,
                                                                       self.batch_size)

        # save data       
        self.save_processed_data(train_X, test_X, train_Y, test_Y,
                                 param_normalizer, pressure_normalizer,
                                 uptake_normalizer, self.host_encoder, self.guest_encoder) 

        return train_dataset, test_dataset           



    
          
