# [SET KERAS BACKEND]
import os 
os.environ["KERAS_BACKEND"] = "torch"

# [IMPORT LIBRARIES]
import pandas as pd

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from NISTADS.commons.utils.preprocessing.pipeline import ProcessPipeline
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [RUN MAIN]
###############################################################################
if __name__ == '__main__':

    # 1. [LOAD AND ENRICH DATA]
    #--------------------------------------------------------------------------     
    # load data from csv, retrieve and merge molecular properties 
    pipeline = ProcessPipeline()
    preprocessed_data = pipeline.materials_dataset_pipeline()

    pass

    
    

   

  

    

    

