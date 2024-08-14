import numpy as np
import tensorflow as tf
import keras

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger
             
        
# [CUSTOM DATA GENERATOR FOR TRAINING]
###############################################################################
# Generate and preprocess input and output for the machine learning model and build
# a tensor dataset with prefetching and batching
###############################################################################
class DataGenerator():

    def __init__(self):              
        
        self.img_shape = CONFIG["model"]["IMG_SHAPE"]       
        self.normalization = CONFIG["dataset"]["IMG_NORMALIZE"]
        self.augmentation = CONFIG["dataset"]["IMG_AUGMENT"] 
        
    
    # load and preprocess a single image
    #--------------------------------------------------------------------------
    def load_image(self, path):

        '''
        Loads and preprocesses a single image.

        Keyword arguments:
            path (str): The path to the image file.

        Returns:
            rgb_image (tf.Tensor): The preprocessed RGB image tensor.

        '''
        image = tf.io.read_file(path)
        rgb_image = tf.image.decode_image(image, channels=3, expand_animations=False)        
        rgb_image = tf.image.resize(rgb_image, self.img_shape[:-1])
        if self.augmentation:
            rgb_image = self.image_augmentation(rgb_image)
        if self.normalization:
            rgb_image = rgb_image/255.0 

        return rgb_image, rgb_image     
    

    # define method perform data augmentation    
    #--------------------------------------------------------------------------
    def image_augmentation(self, image):    
           
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image) 

        return image
              
    # effectively build the tf.dataset and apply preprocessing, batching and prefetching
    #------------------------------------------------------------------------------
    def build_tensor_dataset(self, data, batch_size=None, buffer_size=tf.data.AUTOTUNE):

        '''
        Builds a TensorFlow dataset and applies preprocessing, batching, and prefetching.

        Keyword arguments:
            data (list): A list of image file paths.
            buffer_size (int): The buffer size for shuffling and prefetching (default is tf.data.AUTOTUNE).

        Returns:
            dataset (tf.data.Dataset): The prepared TensorFlow dataset.

        '''
        num_samples = len(data) 
        if batch_size is None:
            batch_size = CONFIG["training"]["BATCH_SIZE"]

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=num_samples)          
        dataset = dataset.map(self.load_image, num_parallel_calls=buffer_size)        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=buffer_size)

        return dataset

    
# wrapper function to run the data pipeline from raw inputs to tensor dataset
###############################################################################
def training_data_pipeline(train_data, validation_data, batch_size=None):    
        
        generator = DataGenerator()

        train_dataset = generator.build_tensor_dataset(train_data, batch_size=batch_size)
        validation_dataset = generator.build_tensor_dataset(validation_data, batch_size=batch_size)        
        for x, y in train_dataset.take(1):
            logger.debug(f'X batch shape is: {x.shape}')  
            logger.debug(f'Y batch shape is: {y.shape}') 

        return train_dataset, validation_dataset



            
