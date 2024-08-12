import os
import pandas as pd
import tensorflow as tf
from keras.utils import plot_model
from tqdm import tqdm
tqdm.pandas()

# [SETTING WARNINGS]
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# [IMPORT CUSTOM MODULES]
from SCADS.commons.utils.preprocessing import PreProcessPipeline
from SCADS.commons.utils.models import ModelTraining, SCADSModel, model_savefolder
from SCADS.commons.utils.callbacks import RealTimeHistory
from SCADS.commons.pathfinder import DATA_PATH, CHECKPOINT_PATH
import SCADS.commons.configurations as cnf


# [RUN MAIN]
if __name__ == '__main__':

    # 1. [LOAD AND PREPROCESS DATA]
    #--------------------------------------------------------------------------
    model_folder_path, model_folder_name = model_savefolder(CHECKPOINT_PATH, 'SCADS')
    pp_path = os.path.join(model_folder_path, 'preprocessing')
    os.mkdir(pp_path) if not os.path.exists(pp_path) else None

    # load data from .csv files    
    file_loc = os.path.join(DATA_PATH, 'SCADS_dataset.csv') 
    df_adsorption = pd.read_csv(file_loc, sep=';', encoding = 'utf-8')

    # start preprocessing pipeline
    pipeline = PreProcessPipeline(cnf.MAX_PRESSURE, cnf.MAX_UPTAKE, cnf.PAD_VALUE,
                                   cnf.PAD_LENGTH, cnf.BATCH_SIZE, cnf.TEST_SIZE,
                                   cnf.SPLIT_SEED, pp_path)
    
    train_dataset, test_dataset = pipeline.run_pipeline(df_adsorption)    

    # 2. [BUILD SCADS MODEL]
    #--------------------------------------------------------------------------
    # print report

    print('SCADS training report\n')
    print('--------------------------------------------------------------------')    
    print(f'Batch size:              {cnf.BATCH_SIZE}')
    print(f'Epochs:                  {cnf.EPOCHS}\n') 
    
    # initialize training device    
    trainer = ModelTraining(device=cnf.ML_DEVICE, use_mixed_precision=cnf.USE_MIXED_PRECISION,
                            seed=cnf.SEED) 

    # determine number of classes and features, then initialize and build the model    
    num_features = len(pipeline.processor.PARAMETERS)   
    unique_adsorbents = len(pipeline.host_encoder.categories_[0])
    unique_sorbates = len(pipeline.guest_encoder.categories_[0]) 
    modelworker = SCADSModel(cnf.LEARNING_RATE, num_features, cnf.PAD_LENGTH, 
                            cnf.PAD_VALUE, unique_adsorbents, unique_sorbates, 
                            cnf.EMBEDDING_DIMS, cnf.SEED, XLA_acceleration=cnf.XLA_ACCELERATION)

    model = modelworker.get_model(summary=True) 

    # generate graphviz plot for the model layout    
    if cnf.GENERATE_MODEL_GRAPH==True:
        plot_path = os.path.join(model_folder_path, 'model_layout.png')       
        plot_model(model, to_file=plot_path, show_shapes=True, 
                show_layer_names=True, show_layer_activations=True, 
                expand_nested=True, rankdir='TB', dpi=400)

    # 3. [TRAIN SCADS MODEL]
    #--------------------------------------------------------------------------
    # Setting callbacks and training routine for the XRAY captioning model. 
    # to visualize tensorboard report, use command prompt on the model folder and 
    # upon activating environment, use the bash command: 
    # python -m tensorboard.main --logdir tensorboard/

    # initialize real time plot callback     
    RTH_callback = RealTimeHistory(model_folder_path, validation=True)

    # initialize tensorboard    
    if cnf.USE_TENSORBOARD == True:
        log_path = os.path.join(model_folder_path, 'tensorboard')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)
        callbacks = [RTH_callback, tensorboard_callback]    
    else:    
        callbacks = [RTH_callback]

    # define and execute training loop, then save the model weights at end
    #------------------------------------------------------------------------------
    multiprocessing = cnf.NUM_PROCESSORS > 1
    training = model.fit(train_dataset, validation_data=test_dataset,  
                        epochs=cnf.EPOCHS, verbose=1, shuffle=True, callbacks=callbacks, 
                        workers=cnf.NUM_PROCESSORS,
                        use_multiprocessing=multiprocessing)

    model_files_path = os.path.join(model_folder_path, 'model')
    model.save(model_files_path, save_format='tf')

    print(f'Training session is over. Model has been saved in folder {model_folder_name}')
  
    # save model data and model parameters in txt files    
    parameters = {'train_samples' : pipeline.num_train_samples,
                  'test_samples' : pipeline.num_test_samples,             
                  'sequence_length' : cnf.PAD_LENGTH,
                  'padding_value' : cnf.PAD_VALUE,
                  'embedding_dimensions' : cnf.EMBEDDING_DIMS,             
                  'batch_size' : cnf.BATCH_SIZE,
                  'learning_rate' : cnf.LEARNING_RATE,
                  'epochs' : cnf.EPOCHS}

    trainer.model_parameters(parameters, model_folder_path)

