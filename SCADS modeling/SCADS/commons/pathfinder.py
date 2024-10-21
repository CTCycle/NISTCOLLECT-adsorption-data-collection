from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
RES_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RES_PATH, 'data')
VALIDATION_PATH = join(RES_PATH, 'validation')
TRAIN_PATH = join(PROJECT_DIR, 'training')
CHECKPOINT_PATH = join(TRAIN_PATH, 'checkpoints')
INFERENCE_PATH = join(PROJECT_DIR, 'inference')



