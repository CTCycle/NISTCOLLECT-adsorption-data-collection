import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger



# [DATASET OPERATIONS]
###############################################################################
class AdsorptionDataset:


    def __init__(self):
        self.drop_cols = ['DOI', 'category', 'tabular_data', 
                              'isotherm_type', 'digitizer', 
                              'articleSource', 'concentrationUnits']
        self.explode_cols = ['pressure', 'adsorbed_amount']

    #--------------------------------------------------------------------------           
    def remove_columns(self, dataframe : pd.DataFrame):

        df_drop = dataframe.drop(columns=self.drop_cols, axis=1)

        return df_drop

    #--------------------------------------------------------------------------           
    def split_by_mixture_complexity(self, dataframe : pd.DataFrame):   

        '''
        Splits the dataframe into two groups based on the number of adsorbates.
        
        Keywords arguments:
            dataframe (DataFrame): the raw adsorption dataset to be processed.

        Returns:
            tuple: A tuple containing two dataframes, one for each group (single_compound, binary_mixture)
        '''

        dataframe['numGuests'] = dataframe['adsorbates'].apply(lambda x : len(x))          
        df_grouped = dataframe.groupby('numGuests')
        single_compound = df_grouped.get_group(1)
        binary_mixture = df_grouped.get_group(2)                
        
        return single_compound, binary_mixture   

    #--------------------------------------------------------------------------
    def process_experiment_data(self, dataframe : pd.DataFrame): 

        '''
        Processes the experimental data contained in the given DataFrame.
        
        This function processes the experimental adsorption data, extracting and 
        transforming relevant information for single-component and binary mixture 
        datasets. Specifically, it extracts adsorbent and adsorbate details, 
        pressure, and adsorption data. The function handles different structures 
        of data based on the number of guest species present.

        Keywords arguments:
            dataframe (DataFrame): The input DataFrame containing experimental data.

        Returns:
            DataFrame: The processed DataFrame with expanded columns for further analysis.

        '''  
        dataframe['adsorbent_ID'] = dataframe['adsorbent'].apply(lambda x : x['hashkey'])      
        dataframe['adsorbent_name'] = dataframe['adsorbent'].apply(lambda x : x['name'])           
        dataframe['adsorbates_ID'] = dataframe['adsorbates'].apply(lambda x : [f['InChIKey'] for f in x])            
        dataframe['adsorbates_name'] = dataframe['adsorbates'].apply(lambda x : [f['name'] for f in x])

        # check if the number of guest species is one (single component dataset)
        if (dataframe['numGuests'] == 1).all():
            dataframe['pressure'] = dataframe['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
            dataframe['adsorbed_amount'] = dataframe['isotherm_data'].apply(lambda x : [f['total_adsorption'] for f in x])
            dataframe['adsorbates_name'] = dataframe['adsorbates'].apply(lambda x : [f['name'] for f in x][0])
            dataframe['composition'] = 1.0 

        # check if the number of guest species is two (binary mixture dataset)
        elif (dataframe['numGuests'] == 2).all():
            data_placeholder = {'composition' : 1.0, 'adsorption': 1.0}
            dataframe['total_pressure'] = dataframe['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
            dataframe['all_species_data'] = dataframe['isotherm_data'].apply(lambda x : [f['species_data'] for f in x])
            dataframe['compound_1'] = dataframe['adsorbates_name'].apply(lambda x : x[0])        
            dataframe['compound_2'] = dataframe['adsorbates_name'].apply(lambda x : x[1] if len(x) > 1 else None)              
            dataframe['compound_1_data'] = dataframe['all_species_data'].apply(lambda x : [f[0] for f in x])               
            dataframe['compound_2_data'] = dataframe['all_species_data'].apply(lambda x : [f[1] if len(f) > 1 else data_placeholder for f in x])
            dataframe['compound_1_composition'] = dataframe['compound_1_data'].apply(lambda x : [f['composition'] for f in x])              
            dataframe['compound_2_composition'] = dataframe['compound_2_data'].apply(lambda x : [f['composition'] for f in x])            
            dataframe['compound_1_pressure'] = dataframe.apply(lambda x: [a * b for a, b in zip(x['compound_1_composition'], x['total_pressure'])], axis=1)             
            dataframe['compound_2_pressure'] = dataframe.apply(lambda x: [a * b for a, b in zip(x['compound_2_composition'], x['total_pressure'])], axis=1)                
            dataframe['compound_1_adsorption'] = dataframe['compound_1_data'].apply(lambda x : [f['adsorption'] for f in x])               
            dataframe['compound_2_adsorption'] = dataframe['compound_2_data'].apply(lambda x : [f['adsorption'] for f in x])

        return dataframe           
    
    #--------------------------------------------------------------------------
    def expand_dataset(self, single_component : pd.DataFrame, binary_mixture : pd.DataFrame):

        '''
        Expands the datasets by exploding and dropping columns.

        Keywords arguments:
            single_component (DataFrame): The single-component dataset to be processed.
            binary_mixture (DataFrame): The binary-component dataset to be processed.

        Returns:
            SC_exploded_dataset (DataFrame): The expanded single-component dataset.
            BN_exploded_dataset (DataFrame): The expanded binary-component dataset.
            
        '''       
        # processing and exploding data for single component dataset
        explode_cols = ['pressure', 'adsorbed_amount']
        drop_columns = ['date', 'adsorbent', 'adsorbates', 
                        'isotherm_data', 'adsorbent_ID', 'adsorbates_ID']
                
        SC_dataset = single_component.explode(explode_cols)
        SC_dataset[explode_cols] = SC_dataset[explode_cols].astype('float32')
        SC_dataset.reset_index(inplace=True, drop=True)       
        SC_dataset = SC_dataset.drop(columns=drop_columns, axis=1)
        SC_dataset.dropna(inplace=True)
                 
        # processing and exploding data for binary mixture dataset
        explode_cols = ['compound_1_pressure', 'compound_2_pressure',
                        'compound_1_adsorption', 'compound_2_adsorption',
                        'compound_1_composition', 'compound_2_composition']
        drop_columns.extend(['adsorbates_name', 'all_species_data', 'compound_1_data', 
                             'compound_2_data', 'adsorbent_ID', 'adsorbates_ID'])        

        BM_dataset = binary_mixture.explode(explode_cols)
        BM_dataset[explode_cols] = BM_dataset[explode_cols].astype('float32')       
        BM_dataset.reset_index(inplace=True, drop=True)        
        BM_dataset = BM_dataset.drop(columns = drop_columns)
        BM_dataset.dropna(inplace=True)  
     
        
        return SC_dataset, BM_dataset



# [DATASET OPERATIONS]
###############################################################################
class AdsorptionDatasetPreparation:    
    
    def __init__(self):
        
        self.datamanager = AdsorptionDataset()     

    #--------------------------------------------------------------------------
    def prepare_dataset(self, dataframe):  

        drop_data = self.datamanager.remove_columns(dataframe) 
        single_component, binary_mixture = self.datamanager.split_by_mixture_complexity(drop_data) 
        single_component = self.datamanager.process_experiment_data(single_component)
        binary_mixture = self.datamanager.process_experiment_data(binary_mixture)
        single_component, binary_mixture = self.datamanager.expand_dataset(single_component, binary_mixture)
        self.save_adsorption_datasets(single_component, binary_mixture) 

        return single_component, binary_mixture 

    #--------------------------------------------------------------------------
    def save_adsorption_datasets(self, single_component : pd.DataFrame, 
                                 binary_mixture : pd.DataFrame): 
        
        file_loc = os.path.join(DATA_PATH, 'single_component_adsorption.csv') 
        single_component.to_csv(file_loc, index=False, sep=';', encoding='utf-8')
        file_loc = os.path.join(DATA_PATH, 'binary_mixture_adsorption.csv') 
        binary_mixture.to_csv(file_loc, index=False, sep=';', encoding='utf-8')         
        