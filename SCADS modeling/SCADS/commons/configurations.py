# Advanced settings for training 
#------------------------------------------------------------------------------
USE_MIXED_PRECISION = False
USE_TENSORBOARD = False
XLA_ACCELERATION = False
ML_DEVICE = 'GPU'
NUM_PROCESSORS = 6

# Settings for training routine
#------------------------------------------------------------------------------
EPOCHS = 2000
LEARNING_RATE = 0.0001
BATCH_SIZE = 1024

# Model settings
#------------------------------------------------------------------------------
EMBEDDING_DIMS = 256
GENERATE_MODEL_GRAPH = True

# Settings for training data 
#------------------------------------------------------------------------------
NUM_SAMPLES = 30000 #set higher than available samples to take all of them
TEST_SIZE = 0.1
PAD_LENGTH = 40
PAD_VALUE = -1
MIN_POINTS = 6
MAX_PRESSURE = 10e06
MAX_UPTAKE = 10

# General settings
#------------------------------------------------------------------------------
SEED = 54
SPLIT_SEED = 425