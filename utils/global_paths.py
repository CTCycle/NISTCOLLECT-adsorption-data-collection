import os

# Define paths for the script
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
proc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processing')

# create folders if they do not exist
#------------------------------------------------------------------------------
os.mkdir(data_path) if not os.path.exists(data_path) else None
os.mkdir(model_path) if not os.path.exists(model_path) else None
os.mkdir(proc_path) if not os.path.exists(proc_path) else None


