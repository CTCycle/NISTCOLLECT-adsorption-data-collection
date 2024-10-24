import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()   
    
          
# [DATA VALIDATION]
#==============================================================================
# 
#==============================================================================
class DataValidation:
    
    def __init__(self):
        
        self.parameters = ['temperature', 'mol_weight', 'complexity', 'covalent_units', 
                           'H_acceptors', 'H_donors', 'heavy_atoms']
        self.categoricals  = ['adsorbent_name', 'adsorbate_name'] 
        self.sequences  = ['pressure_in_Pascal', 'uptake_in_mol_g']        

    #--------------------------------------------------------------------------
    def check_missing_values(self, dataset):

        '''
        Checks for missing values in each column of the dataset 
        and prints a report of the findings.

        Keyword arguments:
            dataset (DataFrame): The dataset to be checked for missing values.

        Returns:
            Series: A pandas Series object where the index corresponds to the column names of the dataset and 
                    the values are the counts of missing values in each column.

        '''
        missing_values = dataset.isnull().sum()
        if missing_values.any():
            print(f'{len(missing_values)} columns have missing values:')
            print(missing_values[missing_values > 0])            
        else:
            print('No columns with missing values\n')

        return missing_values         

    #--------------------------------------------------------------------------
    def plot_histograms(self, dataset, path, exclude_cols=[]):

        '''
        Generates histograms for specified columns in a dataset and saves 
        them as JPEG files to a given directory. This function iterates through 
        a list of columns, generating a histogram for each. Each histogram is 
        saved with a filename indicating the column it represents.

        Keyword arguments:
            dataset (DataFrame): The dataset from which to generate histograms.
            path (str): The directory path where the histogram images will be saved.

        Return:
            None

        '''
        histogram_cols = self.parameters + self.sequences        
        for column in tqdm(histogram_cols):
            if column in exclude_cols:
                continue
            plot_path = os.path.join(path, f'{column}_histogram.jpeg')
            values = dataset[column].values
            if column in self.sequences:
                values = dataset[column].explode(column).values                          
            plt.figure(figsize=(16,18))  
            plt.hist(values, bins=20, color='skyblue', edgecolor='black')
            plt.title(f'histogram_{column}')
            plt.xlabel(column, fontsize=14)
            plt.ylabel('Frequency', fontsize=14)                                       
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches='tight', format='jpeg', dpi=300)
            plt.show(block=False)
            plt.close()

    #--------------------------------------------------------------------------
    def features_comparison(self, train_X, test_X, train_Y, test_Y):

        '''
        Compares the statistical properties (mean and standard deviation) of training and testing 
        datasets for both features and labels.

        Keyword arguments:
            train_X (DataFrame): The training set features.
            test_X (DataFrame): The testing set features.
            train_Y (Series/DataFrame): The training set labels.
            test_Y (Series/DataFrame): The testing set labels.

        Returns:
            dict: A dictionary containing the mean and standard deviation differences for each column in the features, 
                and for the labels, under the key 'Y'. Each entry is a dictionary with keys 'mean_diff' and 'std_diff'.

        '''
        stats = {}  
        features_cols = self.parameters + [self.sequences[0]]   
        
        for col in features_cols:
            if col == self.sequences[0]:
                train_X[col] = train_X[col].explode(col) 
                test_X[col] = test_X[col].explode(col) 
            train_mean, test_mean = train_X[col].mean(), test_X[col].mean()
            train_std, test_std = train_X[col].std(), test_X[col].std()
            mean_diff = abs(train_mean - test_mean)
            std_diff = abs(train_std - test_std)
            stats[col] = [mean_diff, std_diff]
            stats[col] = {'mean_diff': mean_diff, 'std_diff': std_diff}

        train_Y, test_Y = train_Y.explode(), test_Y.explode()
        train_mean_Y, test_mean_Y = train_Y.mean(), test_Y.mean()
        train_std_Y, test_std_Y = train_Y.std(), test_Y.std()
        mean_diff_Y = abs(train_mean_Y - test_mean_Y)
        std_diff_Y = abs(train_std_Y - test_std_Y)
        stats['Y'] = {'mean_diff': mean_diff_Y, 'std_diff': std_diff_Y}

        return stats
    
    #--------------------------------------------------------------------------
    def data_split_validation(self, dataset, test_size, range_val):

        '''
        Evaluates various train-test splits to find the one where the difference in statistical properties (mean and standard deviation) 
        between the training and testing sets is minimized for both features and labels.

        Keyword arguments:
            dataset (DataFrame): The dataset to be split into training and testing sets.
            test_size (float): The proportion of the dataset to include in the test split.
            range_val (int): The number of different splits to evaluate.

        Returns:
            tuple: Contains the minimum difference found across all splits, the seed for the best split, and the statistics 
                for this split. The statistics are a dictionary with keys for each feature and 'Y' for labels, 
                where each entry is a dictionary with 'mean_diff' and 'std_diff'.
        '''
        inputs = dataset[[x for x in dataset.columns if x != self.sequences[1]]]
        labels = dataset[self.sequences[1]]

        min_diff = float('inf')
        best_i = None
        best_stats = None
        for i in tqdm(range(range_val)):
            train_X, test_X, train_Y, test_Y = train_test_split(inputs, labels, test_size=test_size, 
                                                                random_state=i+1, shuffle=True, 
                                                                stratify=None)
            # function call to compare columns by mean and standard deviation
            stats = self.features_comparison(train_X, test_X, train_Y, test_Y)
            # Calculate total difference for this split
            total_diff = sum([sum(values.values()) for key, values in stats.items()])
            # Update values only if the difference is lower in this iteration
            if total_diff < min_diff:
                min_diff = total_diff
                best_seed = i + 1
                best_stats = stats        

        return min_diff, best_seed, best_stats
    
# [MODEL VALIDATION]
#============================================================================== 
class ModelValidation: 

    def __init__(self, model):

        self.model = model     
    
    # comparison of data distribution using statistical methods 
    #--------------------------------------------------------------------------     
    def visualize_predictions(self, X, Y_real, Y_predicted, name='Series', plot_path=None):       

        fig_path = os.path.join(plot_path, f'Visual_validation_{name}.jpeg')
        fig, axs = plt.subplots(2, 2)       
        axs[0, 0].plot(X[0], Y_predicted[0], label='Predicted')
        axs[0, 0].plot(X[0], Y_real[0], label='Actual')         
        axs[0, 1].plot(X[1], Y_predicted[1])
        axs[0, 1].plot(X[1], Y_real[1])        
        axs[1, 0].plot(X[2], Y_predicted[2])
        axs[1, 0].plot(X[2], Y_real[2])        
        axs[1, 1].plot(X[3], Y_predicted[3])
        axs[1, 1].plot(X[3], Y_real[3])        
        for ax in axs.flat:
            ax.set_ylabel('mol/g adsorbed')
            ax.set_xlabel('pressure (Pa)')
        fig.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches='tight', format='jpeg', dpi=400) 
        plt.show()       
        plt.close()       