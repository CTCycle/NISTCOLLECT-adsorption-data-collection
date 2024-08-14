import os
import sys
import pandas as pd
import numpy as np
import pickle 

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# add parent folder path to the namespace
#------------------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

# import modules and classes
#------------------------------------------------------------------------------
from NISTADS.commons.utils.preprocessing import PreProcessing
from NISTADS.commons.utils.inference import Inference
from NISTADS.commons.utils.validation import ModelValidation
from NISTADS.commons.pathfinder import DATA_PATH, CHECKPOINT_PATH, INFERENCE_PATH
import NISTADS.commons.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':    

    # 1. [LOAD MODEL AND DATA]  

    # load the model for inference and print summary
    #------------------------------------------------------------------------------
    inference = Inference(cnf.SEED) 
    model, parameters = inference.load_pretrained_model(CHECKPOINT_PATH)
    NLP_PATH = inference.folder_path
    model.summary(expand_nested=True)

    file_loc = os.path.join(INFERENCE_PATH, 'adsorption_inputs.csv') 
    df_predictions = pd.read_csv(file_loc, sep=';', encoding='utf-8')
    file_loc = os.path.join(INFERENCE_PATH, 'adsorbates_dataset.csv') 
    df_adsorbates = pd.read_csv(file_loc, sep=';', encoding='utf-8')


