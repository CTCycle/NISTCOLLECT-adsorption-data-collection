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
    SCADS_dataset = pipeline.run_preprocessing_pipeline()

    pass
    # pipeline = PreProcessPipeline(cnf.MAX_PRESSURE, cnf.MAX_UPTAKE, cnf.PAD_VALUE,
    #                                cnf.PAD_LENGTH, cnf.BATCH_SIZE, cnf.TEST_SIZE,
    #                                cnf.SPLIT_SEED, pp_path)
    
    # train_dataset, test_dataset = pipeline.run_pipeline(df_adsorption)

    
    # sample_size = CONFIG["dataset"]["SAMPLE_SIZE"] 
    # file_loc = os.path.join(DATA_PATH, DATASET_NAME) 
    # dataset = pd.read_csv(file_loc, encoding='utf-8', sep=';', low_memory=False)
    # dataset = get_images_from_dataset(IMG_DATA_PATH, dataset, sample_size=sample_size)

    # # split data
    # logger.info('Preparing dataset of images based on splitting size')  
    # splitter = DatasetSplit(dataset)     
    # train_data, validation_data = splitter.split_data()       

    # # 2. [PREPROCESS DATA]
    # #--------------------------------------------------------------------------
    # # preprocess text corpus using pretrained distillBERT tokenizer. Text is tokenized
    # # using subwords and these are eventually mapped to integer indexes
    # logger.info('Loading distilBERT tokenizer and apply tokenization')     
    # tokenization = BERTokenizer()    
    # train_data, validation_data = tokenization.BERT_tokenization(train_data, validation_data)
    # tokenizer = tokenization.tokenizer
    # vocab_size = tokenization.vocab_size

    # # save preprocessed data
    # dataserializer = DataSerializer()
    # dataserializer.save_preprocessed_data(train_data, validation_data, DATA_PATH)
    # logger.info(f'Data has been succesfully preprocessed and saved in {DATA_PATH}') 
    

   

  

    

    

