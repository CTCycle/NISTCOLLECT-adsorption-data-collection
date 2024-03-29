import pandas as pd
from tqdm import tqdm
tqdm.pandas()

#------------------------------------------------------------------------------
def pressure_converter(type, p_val):

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

#------------------------------------------------------------------------------
def uptake_converter(q_unit, q_val, mol_weight):

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
def add_guest_properties(df_isotherms, df_adsorbates):

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

#------------------------------------------------------------------------------
def remove_leading_zeros(self, sequence_A, sequence_B):

    # Find the index of the first non-zero element or get the last index if all are zeros
    first_non_zero_index_A = next((i for i, x in enumerate(sequence_A) if x != 0), len(sequence_A) - 1)
    first_non_zero_index_B = next((i for i, x in enumerate(sequence_B) if x != 0), len(sequence_B) - 1)
            
    # Ensure to remove leading zeros except one, for both sequences
    processed_seq_A = sequence_A[max(0, first_non_zero_index_A - 1):]
    processed_seq_B = sequence_B[max(0, first_non_zero_index_B - 1):]        
        
    return processed_seq_A, processed_seq_B  

