import os
import sys
import cv2
import json
import pandas as pd
import keras
from datetime import datetime
import tensorflow as tf

from NISTADS.commons.constants import CONFIG, DATA_PATH, CHECKPOINT_PATH
from NISTADS.commons.logger import logger


# get the path of multiple images from a given directory
###############################################################################
def get_datasets():    
    
    adsorption_path = os.path.join(DATA_PATH, 'single_component_adsorption.csv') 
    adsorption_data = pd.read_csv(adsorption_path, encoding='utf-8', sep=';')     
    guest_path = os.path.join(DATA_PATH, 'guests_dataset.csv') 
    guest_properties = pd.read_csv(guest_path, encoding='utf-8', sep=';')
    host_path = os.path.join(DATA_PATH, 'host_dataset.csv') 
    host_properties = pd.read_csv(host_path, encoding='utf-8', sep=';')      

    return adsorption_data, guest_properties, host_properties 


# [DATA SERIALIZATION]
###############################################################################
class DataSerializer:

    def __init__(self):        
        pass
            
    

    # ...
    #--------------------------------------------------------------------------
    def save_preprocessed_data(self, train_data, validation_data, path):  
         
        combined_data = {'train': train_data, 'validation': validation_data}
        json_path = os.path.join(path, 'data', 'input_data.json')
        with open(json_path, 'w') as json_file:
            json.dump(combined_data, json_file)
            logger.debug(f'Preprocessed data has been saved at {json_path}')

    # ...
    #--------------------------------------------------------------------------
    def load_preprocessed_data(self, path):

        json_path = os.path.join(path, 'data', 'input_data.json')    
        if not os.path.exists(json_path):
            logger.error(f'The file {json_path} does not exist.')
            
        with open(json_path, 'r') as json_file:
            combined_data = json.load(json_file)
            logger.debug(f'Preprocessed data has been loaded from {json_path}')
        
        train_data = combined_data.get('train', None)
        validation_data = combined_data.get('validation', None)

        # reconstruct images path        
        train_data = [os.path.join(IMG_DATA_PATH, os.path.basename(x))
                      for x in train_data if os.path.basename(x) in os.listdir(IMG_DATA_PATH)]   
        validation_data = [os.path.join(IMG_DATA_PATH, os.path.basename(x))
                          for x in validation_data if os.path.basename(x) in os.listdir(IMG_DATA_PATH)]     
        
        return train_data, validation_data  

    
# [MODEL SERIALIZATION]
###############################################################################
class ModelSerializer:

    def __init__(self):
        self.model_name = 'FeXT'

    # function to create a folder where to save model checkpoints
    #--------------------------------------------------------------------------
    def create_checkpoint_folder(self):

        '''
        Creates a folder with the current date and time to save the model.

        Keyword arguments:
            None

        Returns:
            str: A string containing the path of the folder where the model will be saved.
        
        '''        
        today_datetime = datetime.now().strftime('%Y%m%dT%H%M%S')        
        checkpoint_folder_path = os.path.join(CHECKPOINT_PATH, f'{self.model_name}_{today_datetime}')         
        os.makedirs(checkpoint_folder_path, exist_ok=True)        
        os.makedirs(os.path.join(checkpoint_folder_path, 'data'), exist_ok=True)
        logger.debug(f'Created checkpoint folder at {checkpoint_folder_path}')
        
        return checkpoint_folder_path    

    #--------------------------------------------------------------------------
    def save_pretrained_model(self, model : keras.Model, path):

        model_files_path = os.path.join(path, 'saved_model.keras')
        model.save(model_files_path)
        logger.info(f'Training session is over. Model has been saved in folder {path}')

    #--------------------------------------------------------------------------
    def save_model_parameters(self, path, parameters_dict : dict):

        '''
        Saves the model parameters to a JSON file. The parameters are provided 
        as a dictionary and are written to a file named 'model_parameters.json' 
        in the specified directory.

        Keyword arguments:
            parameters_dict (dict): A dictionary containing the parameters to be saved.
            path (str): The directory path where the parameters will be saved.

        Returns:
            None  

        '''
        param_path = os.path.join(path, 'model_parameters.json')      
        with open(param_path, 'w') as f:
            json.dump(parameters_dict, f)
            logger.debug(f'Model parameters have been saved at {path}')

    #--------------------------------------------------------------------------
    def save_model_plot(self, model, path):

        if CONFIG["model"]["SAVE_MODEL_PLOT"]:
            logger.debug('Generating model architecture graph')
            plot_path = os.path.join(path, 'model_layout.png')       
            keras.utils.plot_model(model, to_file=plot_path, show_shapes=True, 
                       show_layer_names=True, show_layer_activations=True, 
                       expand_nested=True, rankdir='TB', dpi=400)
            
    #-------------------------------------------------------------------------- 
    def load_pretrained_model(self):

        '''
        Load a pretrained Keras model from the specified directory. If multiple model 
        directories are found, the user is prompted to select one. If only one model 
        directory is found, that model is loaded directly. If a 'model_parameters.json' 
        file is present in the selected directory, the function also loads the model 
        parameters.

        Keyword arguments:
            path (str): The directory path where the pretrained models are stored.
            load_parameters (bool, optional): If True, the function also loads the 
                                            model parameters from a JSON file. 
                                            Default is True.

        Returns:
            model (keras.Model): The loaded Keras model.
            configuration (dict): The loaded model parameters, or None if the parameters file is not found.

        '''  
        # look into checkpoint folder to get pretrained model names      
        model_folders = []
        for entry in os.scandir(CHECKPOINT_PATH):
            if entry.is_dir():
                model_folders.append(entry.name)

        # quit the script if no pretrained models are found 
        if len(model_folders) == 0:
            logger.error('No pretrained model checkpoints in resources')
            sys.exit()

        # select model if multiple checkpoints are available
        if len(model_folders) > 1:
            model_folders.sort()
            index_list = [idx + 1 for idx, item in enumerate(model_folders)]     
            print('Currently available pretrained models:')             
            for i, directory in enumerate(model_folders):
                print(f'{i + 1} - {directory}')                         
            while True:
                try:
                    dir_index = int(input('\nSelect the pretrained model: '))
                    print()
                except ValueError:
                    logger.error('Invalid choice for the pretrained model, asking again')
                    continue
                if dir_index in index_list:
                    break
                else:
                    logger.warning('Model does not exist, please select a valid index')
                    
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[dir_index - 1])

        # load directly the pretrained model if only one is available 
        elif len(model_folders) == 1:
            logger.info('Loading pretrained model directly as only one is available')
            self.loaded_model_folder = os.path.join(CHECKPOINT_PATH, model_folders[0])                 
            
        # effectively load the model using keras builtin method
        NLP_PATH = os.path.join(self.loaded_model_folder, 'saved_model.keras') 
        model = keras.models.load_model(NLP_PATH)
        
        # load configuration data from .json file in checkpoint folder
        config_path = os.path.join(self.loaded_model_folder, 'model_parameters.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                configuration = json.load(f)                   
        else:
            logger.warning('model_parameters.json file not found. Model parameters were not loaded.')
            configuration = None    
            
        return model, configuration