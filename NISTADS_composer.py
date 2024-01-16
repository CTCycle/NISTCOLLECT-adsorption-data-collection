import sys
import art

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# import modules and classes
#------------------------------------------------------------------------------
from modules.components.data_assets import UserOperations

# welcome message
#------------------------------------------------------------------------------
ascii_art = art.text2art('NISTADS COMPOSER')
print(ascii_art)

# [MAIN MENU]
#==============================================================================
# Ask for user prompts
#==============================================================================
user_operations = UserOperations()
operations_menu = {'1': 'Collect Guest-Host data', 
                   '2': 'Collect adsorption isotherm data',                  
                   '3': 'Exit and close'}

while True:
    print('------------------------------------------------------------------------')
    print('MAIN MENU')
    print('------------------------------------------------------------------------')
    op_sel = user_operations.menu_selection(operations_menu)
    print()    
    if op_sel == 1:
        import modules.GH_composer
        del sys.modules['modules.GH_composer']        
    elif op_sel == 2:
        import modules.experiments_composer
        del sys.modules['modules.experiments_composer']
    elif op_sel == 3:
        break



