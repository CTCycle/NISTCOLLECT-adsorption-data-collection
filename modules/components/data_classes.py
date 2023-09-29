import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tqdm import tqdm
tqdm.pandas()

# ...
#==============================================================================
#==============================================================================
#==============================================================================
class UserOperations:
    
    """    
    A class for user operations such as interactions with the console, directories and files
    cleaning and other maintenance operations.
      
    Methods:
        
    menu_selection(menu):  console menu management 
    clear_all_files(path): remove files and directories
   
    """
    
    # print custom menu on console and allows selecting an option
    #==========================================================================
    def menu_selection(self, menu):        
        
        """        
        menu_selection(menu)
        
        Presents a custom menu to the user and returns the selected option.
        
        Keyword arguments:                      
            menu (dict): A dictionary containing the options to be presented to the user. 
                         The keys are integers representing the option numbers, and the 
                         values are strings representing the option descriptions.
        
        Returns:            
            op_sel (int): The selected option number.
        
        """        
        indexes = [idx + 1 for idx, val in enumerate(menu)]
        for key, value in menu.items():
            print('{0} - {1}'.format(key, value))            
        
        print()
        while True:
            try:
                op_sel = int(input('Select the desired operation: '))
            except:
                continue            
            
            while op_sel not in indexes:
                try:
                    op_sel = int(input('Input is not valid, please select a valid option: '))
                except:
                    continue
            break
        
        return op_sel        

# [DATA PREPROCESSING]
#==============================================================================
#==============================================================================
#==============================================================================
class PreProcessing:
    
    """ 
    A class for preprocessing operations in pointwise fashion (with expanded dataset).
    Includes many different methods that can be used in sequence to build a functional
    preprocessing pipeline.
      
    Methods:
        
    __init__(df_SC, df_BN): initializes the class with the single component 
                            and binary mixrture datasets
    
    dataset_splitting():    splits dataset into train, test and validation sets

    """  

    #==========================================================================
    def pressure_converter(self, type, original_P):

        '''
        pressure_converter(type, original_P)

        Converts pressure from the specified unit to Pascals.

        Keyword arguments:
            type (str): The original unit of pressure.
            original_P (int or float): The original pressure value.

        Returns:
            P_value (int): The pressure value converted to Pascals.

        '''           
        P_unit = type
        if P_unit == 'bar':
            P_value = int(original_P * 100000)        
                
        return P_value 

    #==========================================================================
    def uptake_converter(self, q_unit, q_val, mol_weight):

        '''
        uptake_converter(q_unit, q_val, mol_weight)

        Converts the uptake value from the specified unit to moles per gram.

        Keyword arguments:
            q_unit (str):              The original unit of uptake.
            q_val (int or float):      The original uptake value.
            mol_weight (int or float): The molecular weight of the adsorbate.

        Returns:
            Q_value (float): The uptake value converted to moles per gram

        '''
        Q_value = q_val
        if q_unit in ('mmol/g', 'mol/kg'):
            Q_value = q_val/1000 
        elif q_unit == 'mol/g':
            Q_value = q_val
        elif q_unit == 'mmol/kg':
            Q_value = q_val/1000000
        elif q_unit == 'mg/g':
            Q_value = q_val/1000/float(mol_weight)            
        elif q_unit == 'g/g':
            Q_value = (q_val/float(mol_weight))                                   
        elif q_unit == 'wt%':                
            Q_value = ((q_val/100)/float(mol_weight))          
        elif q_unit in ('g Adsorbate / 100g Adsorbent', 'g/100g'):              
            Q_value = ((q_val/100)/float(mol_weight))                            
        elif q_unit in ('ml(STP)/g', 'cm3(STP)/g'):
            Q_value = q_val/22.414      
                
        return Q_value        
        
    #==========================================================================
    def properties_assigner(self, df_isotherms, df_adsorbates):

        df_properties = df_adsorbates[['name', 'complexity', 'atoms', 'mol_weight', 'covalent_units', 'H_acceptors', 'H_donors', 'heavy_atoms']]
        df_properties = df_properties.rename(columns = {'name': 'adsorbates_name'})
        df_isotherms['adsorbates_name'] = df_isotherms['adsorbates_name'].apply(lambda x : x.lower())
        df_properties['adsorbates_name'] = df_properties['adsorbates_name'].apply(lambda x : x.lower())
        df_adsorption = pd.merge(df_isotherms, df_properties, on = 'adsorbates_name', how='left')
        df_adsorption = df_adsorption.dropna().reset_index(drop=True)

        return df_adsorption    
    
    # preprocessing model for tabular data using Keras pipeline    
    #==========================================================================    
    def series_preprocessing(self, series, str_output=False, padding=True, normalization=True,
                             upper=None, pad_value=20, pad_length=10):

        '''
        Normalizes a series of values.
    
        Keyword arguments:
            series (list): A list of values to be normalized
    
        Returns:
            list: A list of normalized values
        
        '''
        processed_series = series 
        if normalization == True:  
            if upper != None:
                processed_series = [x/upper for x in series]
            else:
                max_val = max([float(g) for g in series])
                if max_val == 0.0:
                    max_val = 10e-14
                processed_series = [x/max_val for x in series]
        if padding == True:
            processed_series = pad_sequences([processed_series], maxlen = pad_length, 
                                              value = pad_value, dtype = 'float32', padding = 'post')
            pp_seq = processed_series[0]

        if str_output == True:
            pp_seq = ' '.join([str(x) for x in pp_seq])

        return pp_seq        
        
    #==========================================================================
    def model_savefolder(self, path, model_name):

        '''
        Creates a folder with the current date and time to save the model.
    
        Keyword arguments:
            path (str):       A string containing the path where the folder will be created.
            model_name (str): A string containing the name of the model.
    
        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        raw_today_datetime = str(datetime.now())
        truncated_datetime = raw_today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '').replace('-', '').replace(' ', 'H') 
        model_name = f'{model_name}_{today_datetime}'
        model_savepath = os.path.join(path, model_name)
        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)               
            
        return model_savepath  

# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class AdsorptionDataset:
    
    '''   
    A class collecting methods to build the NIST adsorption dataset from collected
    data. Methods are meant to be used sequentially as they are self-referring 
    (no need to specify input argument in the method). 
      
    Methods:
        
    extract_molecular_properties(df_mol):  retrieve molecular properties from ChemSpyder
    split_by_mixcomplexity():              separates single component and binary mixture measurements
    extract_adsorption_data():             retrieve molecular properties from ChemSpyder    
    
    '''
    def __init__(self, dataframe):
        self.dataframe = dataframe      
    
    #==========================================================================           
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
            self.single_compound = grouped_df.get_group(1)
        except:
            self.single_compound = 'None'
        try:
            self.binary_mixture = grouped_df.get_group(2)
        except:
            self.binary_mixture = 'None'        
        
        return self.single_compound, self.binary_mixture      
      
    
    #==========================================================================
    def extract_adsorption_data(self): 

        '''
        extract_adsorption_data()

        Extracts adsorption data from the single_compound and binary_mixture dataframes.
        This function creates two new dataframes, df_SC and df_BN, as copies of the single_compound and 
        binary_mixture dataframes, respectively. It then extracts various pieces of information from 
        these dataframes, such as the adsorbent ID and name, the adsorbates ID and name, and the pressure 
        and adsorbed amount data. For the binary mixture dataframe, it also calculates the composition and 
        pressure of each compound.

        Returns:
            tuple: A tuple containing two dataframes with the extracted adsorption data (df_SC, df_BN)
        
        '''       
        self.df_SC = self.single_compound.copy()
        self.df_BN = self.binary_mixture.copy()
       
        self.df_SC['adsorbent_ID'] = self.df_SC['adsorbent'].apply(lambda x : x['hashkey'])      
        self.df_SC['adsorbent_name'] = self.df_SC['adsorbent'].apply(lambda x : x['name'])           
        self.df_SC['adsorbates_ID'] = self.df_SC['adsorbates'].apply(lambda x : [f['InChIKey'] for f in x])            
        self.df_SC['adsorbates_name'] = self.df_SC['adsorbates'].apply(lambda x : [f['name'] for f in x][0])
        self.df_SC['pressure'] = self.df_SC['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
        self.df_SC['adsorbed_amount'] = self.df_SC['isotherm_data'].apply(lambda x : [f['total_adsorption'] for f in x])
        self.df_SC['composition'] = 1.0            
                       
        self.df_BN['adsorbent_ID'] = self.df_BN['adsorbent'].apply(lambda x : x['hashkey'])           
        self.df_BN['adsorbent_name'] = self.df_BN['adsorbent'].apply(lambda x : x['name'])               
        self.df_BN['adsorbates_ID'] = self.df_BN['adsorbates'].apply(lambda x : [f['InChIKey'] for f in x])          
        self.df_BN['adsorbates_name'] = self.df_BN['adsorbates'].apply(lambda x : [f['name'] for f in x])         
        self.df_BN['total_pressure'] = self.df_BN['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
        self.df_BN['all_species_data'] = self.df_BN['isotherm_data'].apply(lambda x : [f['species_data'] for f in x])              
        self.df_BN['compound_1_data'] = self.df_BN['all_species_data'].apply(lambda x : [f[0] for f in x])               
        self.df_BN['compound_2_data'] = self.df_BN['all_species_data'].apply(lambda x : [f[0] for f in x])            
        self.df_BN['compound_1_composition'] = self.df_BN['compound_1_data'].apply(lambda x : [f['composition'] for f in x])              
        self.df_BN['compound_2_composition'] = self.df_BN['compound_2_data'].apply(lambda x : [f['composition'] for f in x])            
        self.df_BN['compound_1_pressure'] = self.df_BN.apply(lambda x: [a * b for a, b in zip(x['compound_1_composition'], x['total_pressure'])], axis=1)             
        self.df_BN['compound_2_pressure'] = self.df_BN.apply(lambda x: [a * b for a, b in zip(x['compound_2_composition'], x['total_pressure'])], axis=1)                
        self.df_BN['compound_1_adsorption'] = self.df_BN['compound_1_data'].apply(lambda x : [f['adsorption'] for f in x])               
        self.df_BN['compound_2_adsorption'] = self.df_BN['compound_2_data'].apply(lambda x : [f['adsorption'] for f in x])
                                   
        return self.df_SC, self.df_BN         
    
    
    #==========================================================================
    def dataset_expansion(self):

        '''
        dataset_expansion()

        Expands the datasets by exploding and dropping columns.

        Returns:
            SC_exploded_dataset (DataFrame): The expanded single-component dataset.
            BN_exploded_dataset (DataFrame): The expanded binary-component dataset.

        '''       
        df_SC = self.df_SC.copy()
        df_BN = self.df_BN.copy()       
                         
        explode_cols = ['pressure', 'adsorbed_amount']
        drop_columns = ['DOI', 'date', 'adsorbent', 'concentrationUnits', 
                        'adsorbates', 'isotherm_data', 'adsorbent_ID', 'adsorbates_ID']
        
        SC_exp_dataset = df_SC.explode(explode_cols)
        SC_exp_dataset.reset_index(inplace = True, drop = True)       
        self.SC_exploded_dataset = SC_exp_dataset.drop(columns = drop_columns)        
        df_BN['compound_1'] = df_BN['adsorbates_name'].apply(lambda x : x[0])        
        df_BN['compound_2'] = df_BN['adsorbates_name'].apply(lambda x : x[1])        
        
        explode_cols = ['compound_1_pressure', 'compound_2_pressure',
                        'compound_1_adsorption', 'compound_2_adsorption']
        drop_columns = ['DOI', 'date', 'adsorbates_name', 'adsorbent', 'concentrationUnits',
                        'all_species_data', 'compound_1_data', 'compound_2_data',
                        'adsorbates', 'isotherm_data', 'adsorbent_ID', 'adsorbates_ID']
     
        BN_exp_dataset = df_BN.explode(explode_cols)       
        BN_exp_dataset.reset_index(inplace = True, drop = True)        
        self.BN_exploded_dataset = BN_exp_dataset.drop(columns = drop_columns) 
        
        self.SC_exploded_dataset.dropna(inplace = True)
        self.BN_exploded_dataset.dropna(inplace = True)        
        
        return self.SC_exploded_dataset, self.BN_exploded_dataset       


    
# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class DataStorage: 

    #==========================================================================
    def JSON_serializer(self, object, filename, path, mode='SAVE'):

        if mode == 'SAVE':
            object_json = object.to_json()          
            json_path = os.path.join(path, f'{filename}.json')
            with open(json_path, 'w', encoding = 'utf-8') as f:
                f.write(object_json)
        elif mode == 'LOAD':
            pass




  