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
from utils.preprocessing import add_guest_properties, pressure_converter, uptake_converter, remove_leading_zeros
from utils.pathfinder import DATA_EXP_PATH, DATA_MAT_PATH
import configurations as cnf

# [DEFINE CONSTANTS]
ADS_COL, SORB_COL  = ['adsorbent_name'], ['adsorbates_name'] 
P_COL, Q_COL  = 'pressure_in_Pascal', 'uptake_in_mol_g'
P_UNIT_COL, Q_UNIT_COL  = 'pressureUnits', 'adsorptionUnits' 
VALID_UNITS = ['mmol/g', 'mol/kg', 'mol/g', 'mmol/kg', 'mg/g', 'g/g', 'cm3(STP)/g',
               'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g', 'ml(STP)/g']
                                
PARAMETERS = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                'H_acceptors', 'H_donors', 'heavy_atoms']


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD DATA]
    #--------------------------------------------------------------------------
    filepath = os.path.join(DATA_MAT_PATH, 'adsorbents_dataset.csv')  
    df_adsorbents = pd.read_csv(filepath, sep=';', encoding='utf-8')  
    filepath = os.path.join(DATA_MAT_PATH, 'adsorbates_dataset.csv')  
    df_adsorbates = pd.read_csv(filepath, sep=';', encoding='utf-8')  
    filepath = os.path.join(DATA_EXP_PATH, 'single_component_dataset.csv')  
    df_SC = pd.read_csv(filepath, sep=';', encoding='utf-8')
   
    # 2. [PREPROCESS DATA]
    #--------------------------------------------------------------------------
    # add molecular properties based on PUGCHEM API data    
    print('\nAdding physicochemical properties from guest species dataset')
    dataset = add_guest_properties(df_SC, df_adsorbates)
    dataset = dataset.dropna()

    # filter experiments leaving only valid uptake and pressure units, then convert 
    # pressure and uptake to Pa (pressure) and mol/kg (uptake)    
    print('\nConverting units and filtering bad values\n')

    # filter experiments by pressure and uptake units 
    dataset = dataset[dataset[Q_UNIT_COL].isin(VALID_UNITS)]

    # convert pressures to Pascal
    dataset[P_COL] = dataset.progress_apply(lambda x : pressure_converter(x[P_UNIT_COL], x['pressure']), axis=1)
    # convert uptakes to mol/g
    dataset[Q_COL] = dataset.progress_apply(lambda x : uptake_converter(x[Q_UNIT_COL], x['adsorbed_amount'], 
                                                                        x['mol_weight']), axis=1)

    # further filter the dataset to remove experiments which values are outside desired boundaries, 
    # such as experiments with negative temperature, pressure and uptake values   
    dataset = dataset[dataset['temperature'].astype(int) > 0]
    dataset = dataset[dataset[P_COL].astype(float).between(0.0, cnf.MAX_PRESSURE)]
    dataset = dataset[dataset[Q_COL].astype(float).between(0.0, cnf.MAX_UPTAKE)]

    # Aggregate values using groupby function in order to group the dataset by experiments    
    def join_str(x):
        return ' '.join(x.astype(str))

    aggregate_dict = {'temperature' : 'first',                  
                    'adsorbent_name' : 'first',
                    'adsorbates_name' : 'first',                  
                    'complexity' : 'first',                  
                    'mol_weight' : 'first',
                    'covalent_units' : 'first',
                    'H_acceptors' : 'first',
                    'H_donors' : 'first',
                    'heavy_atoms' : 'first', 
                    'pressure_in_Pascal' : join_str,
                    'uptake_in_mol_g' : join_str}
    
    # group dataset by experiments and drop filename column as it is not necessary
    print('\nGroup by experiment and clean data\n')
    dataset_grouped = dataset.groupby('filename', as_index=False).agg(aggregate_dict)
    dataset_grouped.drop(columns='filename', axis=1, inplace=True)

    # remove series of pressure/uptake with less than X points, drop rows containing nan
    # values and select a subset of samples for training    
    dataset_grouped = dataset_grouped[~dataset_grouped[P_COL].apply(lambda x: all(elem == 0 for elem in x))]
    dataset_grouped = dataset_grouped[dataset_grouped[P_COL].apply(lambda x: len(x)) >= cnf.MIN_POINTS]
    dataset_grouped = dataset_grouped.dropna()
    total_experiments = dataset_grouped.shape[0]

    # preprocess sequences to remove leading 0 values (some experiments may have several
    # zero measurements at the start), make sure that every experiment starts with pressure
    # of 0 Pa and uptake of 0 mol/g (effectively converges to zero)    
    dataset_grouped[[P_COL, Q_COL]] = dataset_grouped.apply(lambda row: 
                    remove_leading_zeros(row[P_COL],
                    row[Q_COL]), axis=1, result_type='expand')

    # save files as csv locally    
    file_loc = os.path.join(DATA_EXP_PATH, 'preprocessed_SC_dataset.csv') 
    dataset_grouped.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
    

