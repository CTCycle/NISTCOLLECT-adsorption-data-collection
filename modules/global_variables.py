import os

# Define paths for the script
#------------------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

# Create folders if they do not exist
#------------------------------------------------------------------------------
if not os.path.exists(data_path):
    os.mkdir(data_path)



