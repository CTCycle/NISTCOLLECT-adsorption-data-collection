import json
from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
RSC_PATH = join(PROJECT_DIR, 'resources')
DATA_PATH = join(RSC_PATH, 'datasets')
PP_DATA_PATH = join(DATA_PATH, 'processed data')
NLP_PATH = join(RSC_PATH, 'NLP models')
VALIDATION_PATH = join(RSC_PATH, 'validation')
CHECKPOINT_PATH = join(RSC_PATH, 'checkpoints')
PREDS_PATH = join(RSC_PATH, 'predictions')
LOGS_PATH = join(PROJECT_DIR, 'resources', 'logs')

CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)