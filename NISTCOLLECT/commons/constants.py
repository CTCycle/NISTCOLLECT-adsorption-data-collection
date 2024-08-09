import json
from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
DATA_PATH = join(PROJECT_DIR, 'resources')
DATA_EXP_PATH = join(DATA_PATH, 'adsorption data')
DATA_MAT_PATH = join(DATA_PATH, 'materials data')
MODEL_PATH = join(PROJECT_DIR, 'experimental', 'model')

LOGS_PATH = join(DATA_PATH, 'logs')

CONFIG_PATH = join(PROJECT_DIR, 'settings', 'configurations.json')
with open(CONFIG_PATH, 'r') as file:
    CONFIG = json.load(file)