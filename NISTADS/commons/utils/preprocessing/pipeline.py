import os
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.utils.preprocessing.aggregation import GuestPropertiesMerge
from NISTADS.commons.utils.preprocessing.conversion import PressureConversion, UptakeConversion
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
class ProcessPipeline:

    def __init__(self):
        properties_path = os.path.join(DATA_PATH, 'guests_dataset.csv') 
        self.properties = pd.read_csv(properties_path, encoding='utf-8', sep=';')
        SCADS_path = os.path.join(DATA_PATH, 'single_component_adsorption.csv') 
        self.SCADS = pd.read_csv(SCADS_path, encoding='utf-8', sep=';')

        self.merger = GuestPropertiesMerge(self.properties, self.SCADS)
        self.P_converter = PressureConversion()
        self.Q_converter = UptakeConversion()

    #--------------------------------------------------------------------------
    def run_preprocessing_pipeline(self):

        processed_data = self.merger.merge_guest_properties()
        processed_data = self.P_converter.convert_data(processed_data)
        processed_data = self.Q_converter.convert_data(processed_data)

        pass
        