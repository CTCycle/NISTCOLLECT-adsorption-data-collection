import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
tqdm.pandas()

    
# [DATA VALIDATION]
###############################################################################
class DataValidation:   

    #--------------------------------------------------------------------------
    def class_distribution(self, df, column, title='', y_label=''):      

        class_counts = df[column].value_counts().reset_index()
        class_counts.columns = ['Class', 'Counts']
        
        plt.figure(figsize=(14, 12))  
        sns.barplot(x='Counts', y='Class', data=class_counts, 
                    color='skyblue', orient='h')
        plt.title(title, fontsize=14)
        plt.xlabel(column, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(rotation=45, ha='center', fontsize=12)
        plt.yticks(rotation=0, ha='right', fontsize=12)
        plt.gca().tick_params(axis='y', which='major', pad=10)  
        plt.tight_layout()  
        plt.show(block=False)

    #--------------------------------------------------------------------------
    def features_boxplot(self, df, columns, title='', x_label='', y_label=''):      

        df_selected = df[columns]        
        plt.figure(figsize=(14, 12)) 
        sns.boxplot(data=df_selected, orient=45, palette='viridis')
        plt.title(title, fontsize=14)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(rotation=45, ha='center', fontsize=14)
        plt.yticks(rotation=0, ha='right', fontsize=14)
        plt.tight_layout()  
        plt.show(block=False)

    #--------------------------------------------------------------------------
    def features_scatterplot(self, df, columns, title='', x_label='', y_label=''):

        df_selected = df[columns]        
        plt.figure(figsize=(14, 12))  
        sns.scatterplot(data=df_selected, x=columns[0], y=columns[1], 
                        color='skyblue', edgecolor='black')
        plt.title(title, fontsize=14)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(rotation=45, ha='center', fontsize=14)
        plt.yticks(rotation=0, ha='right', fontsize=14)
        plt.tight_layout()  
        plt.show(block=False)

    #--------------------------------------------------------------------------
    def DBSCAN_clustering(self, df, min_samples, title='', x_label='', y_label=''):

             
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
        plt.title(title, fontsize=14)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xticks(rotation=45, ha='center', fontsize=14)
        plt.yticks(rotation=0, ha='right', fontsize=14)
        plt.tight_layout()  
        plt.show(block=False)


    

