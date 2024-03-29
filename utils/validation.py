import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
tqdm.pandas()

    
# [DATA VALIDATION]
#==============================================================================
# preprocess adsorption data
#==============================================================================
class DataValidation:   

    #--------------------------------------------------------------------------
    def class_distribution(self, df, column, params):      

        class_counts = df[column].value_counts().reset_index()
        class_counts.columns = ['Class', 'Counts']
        
        plt.figure(figsize=params['figsize'])  
        sns.barplot(x='Counts', y='Class', data=class_counts, 
                    color=params['color'], orient=params['orientation'])
        plt.title(params['title'], fontsize=params['fontsize_title'])
        plt.xlabel(column, fontsize=params['fontsize_labels'])
        plt.ylabel(params['ylabel'], fontsize=params['fontsize_labels'])
        plt.xticks(rotation=params['xticks_rotation'], ha='center', fontsize=params['fontsize_ticks'])
        plt.yticks(rotation=0, ha='right', fontsize=params['fontsize_ticks'])
        plt.gca().tick_params(axis='y', which='major', pad=10)  
        plt.tight_layout()  
        plt.show(block=False)

    #--------------------------------------------------------------------------
    def features_boxplot(self, df, columns, params):      

        df_selected = df[columns]        
        plt.figure(figsize=params['figsize'])   
        sns.boxplot(data=df_selected, orient=params['orientation'], palette=params['palette'])
        plt.title(params['title'], fontsize=params['fontsize_title'])
        plt.xlabel(params['xlabel'], fontsize=params['fontsize_labels'])
        plt.ylabel(params['ylabel'], fontsize=params['fontsize_labels'])
        plt.xticks(rotation=params['xticks_rotation'], ha='center', fontsize=params['fontsize_ticks'])
        plt.yticks(rotation=0, ha='right', fontsize=params['fontsize_ticks'])
        plt.tight_layout()  
        plt.show(block=False)

    #--------------------------------------------------------------------------
    def features_scatterplot(self, df, columns, params):      

        df_selected = df[columns]        
        plt.figure(figsize=params['figsize'])   
        sns.scatterplot(data=df_selected, x=columns[0], y=columns[1], 
                        color=params['color'], edgecolor='black')
        plt.title(params['title'], fontsize=params['fontsize_title'])
        plt.xlabel(params['xlabel'], fontsize=params['fontsize_labels'])
        plt.ylabel(params['ylabel'], fontsize=params['fontsize_labels'])
        plt.xticks(rotation=params['xticks_rotation'], ha='center', fontsize=params['fontsize_ticks'])
        plt.yticks(rotation=0, ha='right', fontsize=params['fontsize_ticks'])
        plt.tight_layout()  
        plt.show(block=False)

    #--------------------------------------------------------------------------
    def DBSCAN_clustering(self, df, min_samples, params):  

             
        X_scaled = StandardScaler().fit_transform(df.values)        
        dbscan = DBSCAN(eps=0.75, min_samples=min_samples, algorithm='ball_tree')        
        dbscan.fit(X_scaled)        
        df['cluster'] = dbscan.labels_
        plt.figure(figsize=(10, 6))
        # Scatter plot of the data points, coloring them based on cluster ID
        # Here, we iterate over the unique cluster IDs and plot each subset of points separately
        clusters = df['cluster'].unique()
        for cluster in clusters:
            # Select data for the current cluster
            cluster_data = df[df['cluster'] == cluster]

            # Choose color black for noise points, which are labeled with -1
            color = 'k' if cluster == -1 else plt.cm.jet(float(cluster) / max(clusters))    
    
        plt.scatter(cluster_data[df.columns[0]], cluster_data[df.columns[1]],
                    s=50, palette=color, label=f'Cluster {cluster}' if cluster != -1 else 'Noise')
        plt.title(params['title'], fontsize=params['fontsize_title'])
        plt.xlabel(params['xlabel'], fontsize=params['fontsize_labels'])
        plt.ylabel(params['ylabel'], fontsize=params['fontsize_labels'])
        plt.xticks(rotation=params['xticks_rotation'], ha='center', fontsize=params['fontsize_ticks'])
        plt.yticks(rotation=0, ha='right', fontsize=params['fontsize_ticks'])
        plt.tight_layout()  
        plt.show(block=False)


    

