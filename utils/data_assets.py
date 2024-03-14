import pandas as pd
import matplotlib.pyplot as plt
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
from tqdm import tqdm
tqdm.pandas()

# [DATASET OPERATIONS]
#==============================================================================
# Methods to perform operation on the built adsorption dataset
#==============================================================================
class AdsorptionDataset:    
    
    def __init__(self, dataframe):
        self.dataframe = dataframe      
    
    #--------------------------------------------------------------------------           
    def split_by_mixcomplexity(self):   

        '''
        split_by_mixcomplexity()

        Splits the dataframe into two groups based on the number of adsorbates.
        This function adds a new column 'num_of_adsorbates' to the dataframe, 
        which contains the number of adsorbates for each row. The dataframe is then 
        grouped by this column and split into two groups: one group with a single adsorbate 
        and another group with two adsorbates. If either of these groups is empty, 
        its value is set to 'None'.

        Returns:
            tuple: A tuple containing two dataframes, one for each group (single_compound, binary_mixture)
        
        '''       
        self.dataframe['num_of_adsorbates'] = self.dataframe['adsorbates'].apply(lambda x : len(x))          
        grouped_df = self.dataframe.groupby('num_of_adsorbates')
        try:
            single_compound = grouped_df.get_group(1)
        except:
            single_compound = pd.DataFrame()
        try:
            binary_mixture = grouped_df.get_group(2)
        except:
            binary_mixture = pd.DataFrame()        
        
        return single_compound, binary_mixture      
      
    
    #--------------------------------------------------------------------------
    def extract_adsorption_data(self, raw_data, num_species=1): 

        '''
        Extracts adsorption data from the single_compound and binary_mixture dataframes.
        This function creates two new dataframes, df_SC and df_BN, as copies of the single_compound and 
        binary_mixture dataframes, respectively. It then extracts various pieces of information from 
        these dataframes, such as the adsorbent ID and name, the adsorbates ID and name, and the pressure 
        and adsorbed amount data. For the binary mixture dataframe, it also calculates the composition and 
        pressure of each compound.

        Returns:
            tuple: A tuple containing two dataframes with the extracted adsorption data (df_SC, df_BN)
        
        '''  
        df_adsorption = raw_data.copy()
        try:
            if num_species==1:                             
                df_adsorption['adsorbent_ID'] = df_adsorption['adsorbent'].apply(lambda x : x['hashkey'])      
                df_adsorption['adsorbent_name'] = df_adsorption['adsorbent'].apply(lambda x : x['name'])           
                df_adsorption['adsorbates_ID'] = df_adsorption['adsorbates'].apply(lambda x : [f['InChIKey'] for f in x])            
                df_adsorption['adsorbates_name'] = df_adsorption['adsorbates'].apply(lambda x : [f['name'] for f in x][0])
                df_adsorption['pressure'] = df_adsorption['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
                df_adsorption['adsorbed_amount'] = df_adsorption['isotherm_data'].apply(lambda x : [f['total_adsorption'] for f in x])
                df_adsorption['composition'] = 1.0 
            elif num_species==2:            
                df_adsorption['adsorbent_ID'] = df_adsorption['adsorbent'].apply(lambda x : x['hashkey'])           
                df_adsorption['adsorbent_name'] = df_adsorption['adsorbent'].apply(lambda x : x['name'])               
                df_adsorption['adsorbates_ID'] = df_adsorption['adsorbates'].apply(lambda x : [f['InChIKey'] for f in x])          
                df_adsorption['adsorbates_name'] = df_adsorption['adsorbates'].apply(lambda x : [f['name'] for f in x])         
                df_adsorption['total_pressure'] = df_adsorption['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
                df_adsorption['all_species_data'] = df_adsorption['isotherm_data'].apply(lambda x : [f['species_data'] for f in x])              
                df_adsorption['compound_1_data'] = df_adsorption['all_species_data'].apply(lambda x : [f[0] for f in x])               
                df_adsorption['compound_2_data'] = df_adsorption['all_species_data'].apply(lambda x : [f[0] for f in x])            
                df_adsorption['compound_1_composition'] = df_adsorption['compound_1_data'].apply(lambda x : [f['composition'] for f in x])              
                df_adsorption['compound_2_composition'] = df_adsorption['compound_2_data'].apply(lambda x : [f['composition'] for f in x])            
                df_adsorption['compound_1_pressure'] = df_adsorption.apply(lambda x: [a * b for a, b in zip(x['compound_1_composition'], x['total_pressure'])], axis=1)             
                df_adsorption['compound_2_pressure'] = df_adsorption.apply(lambda x: [a * b for a, b in zip(x['compound_2_composition'], x['total_pressure'])], axis=1)                
                df_adsorption['compound_1_adsorption'] = df_adsorption['compound_1_data'].apply(lambda x : [f['adsorption'] for f in x])               
                df_adsorption['compound_2_adsorption'] = df_adsorption['compound_2_data'].apply(lambda x : [f['adsorption'] for f in x])
        except:
            pass            
                                   
        return df_adsorption         
    
    #--------------------------------------------------------------------------
    def dataset_expansion(self, df_SC, df_BM):

        '''
        Expands the datasets by exploding and dropping columns.

        Returns:
            SC_exploded_dataset (DataFrame): The expanded single-component dataset.
            BN_exploded_dataset (DataFrame): The expanded binary-component dataset.

        '''       
        df_single = df_SC.copy()
        df_binary = df_BM.copy() 
              
                         
        explode_cols = ['pressure', 'adsorbed_amount']
        drop_columns = ['DOI', 'date', 'adsorbent', 'concentrationUnits', 
                        'adsorbates', 'isotherm_data', 'adsorbent_ID', 'adsorbates_ID']        
        try:
            SC_exp_dataset = df_single.explode(explode_cols)
            SC_exp_dataset[explode_cols] = SC_exp_dataset[explode_cols].astype('float32')   
            SC_exp_dataset.reset_index(inplace=True, drop=True)       
            SC_exploded_dataset = SC_exp_dataset.drop(columns=drop_columns)
        except:
            SC_exploded_dataset = pd.DataFrame()           
        
        explode_cols = ['compound_1_pressure', 'compound_2_pressure',
                        'compound_1_adsorption', 'compound_2_adsorption']
        drop_columns = ['DOI', 'date', 'adsorbates_name', 'adsorbent', 'concentrationUnits',
                        'all_species_data', 'compound_1_data', 'compound_2_data',
                        'adsorbates', 'isotherm_data', 'adsorbent_ID', 'adsorbates_ID']        
        try:
            df_binary['compound_1'] = df_binary['adsorbates_name'].apply(lambda x : x[0])        
            df_binary['compound_2'] = df_binary['adsorbates_name'].apply(lambda x : x[1])        
            BM_exp_dataset = df_binary.explode(explode_cols)
            BM_exp_dataset[explode_cols] = BM_exp_dataset[explode_cols].astype('float32')       
            BM_exp_dataset.reset_index(inplace = True, drop = True)        
            BM_exploded_dataset = BM_exp_dataset.drop(columns = drop_columns)
        except:
            BM_exploded_dataset = pd.DataFrame() 
            
        SC_exploded_dataset.dropna(inplace = True)
        BM_exploded_dataset.dropna(inplace = True)        
        
        return SC_exploded_dataset, BM_exploded_dataset



# [DATA VALIDATION]
#==============================================================================
# preprocess adsorption data
#==============================================================================
class DataValidation:   

    def group_distribution(self, df, column):

        # Grouping the DataFrame by 'group_column' and plotting a histogram of the 'value' column
        grouped_df = df.groupby(by=column)
        plt.figure(figsize=(14, 12))

        # Iterate over groups and plot histogram for each group
        for group_name, group_data in grouped_df:
            plt.hist(group_data[column], bins=10, alpha=0.5, label=group_name)

        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Value Grouped by Group Column')
        plt.legend()
        plt.tight_layout()       
        plt.show(block=False)


# [DATA PREPROCESSING]
#==============================================================================
# preprocess adsorption data
#==============================================================================
class PreProcessing:   


    def __init__(self):
        self.valid_units = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 
                            'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g', 
                            'cm3(STP)/g']
        self.parameters = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                         'H_acceptors', 'H_donors', 'heavy_atoms']
        self.ads_col, self.sorb_col  = ['adsorbent_name'], ['adsorbates_name'] 
        self.P_col, self.Q_col  = 'pressure_in_Pascal', 'uptake_in_mol_g'
        self.P_unit_col, self.Q_unit_col  = 'pressureUnits', 'adsorptionUnits'   

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
        on the 'adsorbates_name' column, assigns properties to each adsorbate, and returns a new
        DataFrame containing the merged data with assigned properties.

        Keyword Arguments:
            df_isotherms (pandas DataFrame): A DataFrame containing isotherm data.
            df_adsorbates (pandas DataFrame): A DataFrame containing adsorbate properties.

        Returns:
            df_adsorption (pandas DataFrame): A DataFrame containing merged isotherm data
                                              with assigned adsorbate properties.

        '''
        df_isotherms['adsorbates_name'] = df_isotherms['adsorbates_name'].str.lower()
        df_adsorbates['adsorbates_name'] = df_adsorbates['name'].str.lower()        
        df_properties = df_adsorbates[['adsorbates_name', 'complexity', 'atoms', 'mol_weight', 
                                        'covalent_units', 'H_acceptors', 'H_donors', 'heavy_atoms']]        
        df_adsorption = pd.merge(df_isotherms, df_properties, on='adsorbates_name', how='inner')

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
        train_X[['adsorbates_name']] = self.guest_encoder.fit_transform(train_X[['adsorbates_name']])
        test_X[['adsorbent_name']] = self.host_encoder.transform(test_X[['adsorbent_name']])
        test_X[['adsorbates_name']] = self.guest_encoder.transform(test_X[['adsorbates_name']])

        return train_X, test_X    
    
    #--------------------------------------------------------------------------  
    def sequence_padding(self, dataset, column, pad_value=-1, pad_length=50):

        '''
        Normalizes a series of values.
    
        Keyword arguments:
            series (list): A list of values to be normalized
    
        Returns:
            list: A list of normalized values
        
        '''        
        dataset[column] = preprocessing.sequence.pad_sequences(dataset[column], 
                                                               maxlen=pad_length, 
                                                               value=pad_value, 
                                                               dtype='float32', 
                                                               padding='post').tolist()           

        return dataset    
        
    
    

# [DATASET OPERATIONS]
#==============================================================================
# Methods to perform operation on the built adsorption dataset
#==============================================================================
class GPT2Model:    
    
    def __init__(self, path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=path)        
        self.model = TFGPT2LMHeadModel.from_pretrained('gpt2', cache_dir=path) 

    def generative_descriptions(self, name):

        # Encode text input to get token ids        
        input_text = f'Provide a brief description of {name}.'        
        inputs = self.tokenizer.encode_plus(input_text, return_tensors="tf", add_special_tokens=True)

        # Extract input_ids and attention_mask from the encoded input
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Generate a sequence from the model
        # Note: Adjust the generate() parameters as needed for your application
        output = self.model.generate(input_ids, attention_mask=attention_mask, 
                                max_length=100, num_return_sequences=1, 
                                no_repeat_ngram_size=2, early_stopping=True)       

        # Decode the output token ids to text
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        print(output_text)
                