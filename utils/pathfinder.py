from os.path import join, dirname, abspath 

PROJECT_DIR = dirname(dirname(abspath(__file__)))
DATA_PATH = join(PROJECT_DIR, 'data')
DATA_EXP_PATH = join(PROJECT_DIR, 'data', 'experiments')
DATA_MAT_PATH = join(PROJECT_DIR, 'data', 'materials')
MODEL_PATH = join(PROJECT_DIR, 'model')
