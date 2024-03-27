import pandas as pd
import numpy as np
import os
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt



def extract_features(directory, flag):

    # Dircetory
    parquet_directory = '/home/nayanika/myelin-h/neuroengineering/Metadata-task3/n_back_data/n_back_data/'

    # List all Parquet files in the directory
    parquet_files = [file for file in os.listdir(directory) if file.endswith('.parquet')]
    parquet_files = sorted(parquet_files)

    list_of_hit_rate = []
    list_of_miss_rate = []
    list_of_FA_rate = []
    list_of_hit_latencies = []
    list_of_FA_latencies = []
    ID_list = []

    # extract features from each file
    for parquet_file in parquet_files:

        # exclude patient 40
        if parquet_file == 'data_42.parquet':
            pass
        else:
            
            file_path = os.path.join(parquet_directory, parquet_file)
            df = pd.read_parquet(file_path)

            df['normalized_time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
            df['group'] = pd.cut(df['i'], bins=[-1, 49, 101, 155], labels=['Group 1', 'Group 2', 'Group 3'])

            desired_event_ids = [1, 2, 3, 4, 5, 6]

            # Count the occurrences of each event ID within each group and fill missing values with 0
            group_counts = df.groupby(['group', 'event_id']).size().unstack(fill_value=0)
            group_counts = group_counts.reindex(columns=desired_event_ids, fill_value=0)

            # print(group_counts)

            hit_rate = (group_counts.loc[:, 1]+ group_counts.loc[:, 3])/group_counts.loc[:, 6]
            FA_rate = (group_counts.loc[:, 2]+ group_counts.loc[:, 4])/group_counts.loc[:, 6]
            miss_rate = (group_counts.loc[:, 5])/group_counts.loc[:, 6]

            # Calculate the time difference between consecutive rows for latencies
            df['time_difference'] = df['normalized_time'].diff()
        
            latencies = df.groupby(['event_id', 'group'])['time_difference'].mean().reset_index()
            hit_latencies = df[df['event_id'].isin([1, 3])].groupby(['group'])['time_difference'].mean().reset_index().set_index('group')
            FA_latencies = df[df['event_id'].isin([2, 4])].groupby(['group'])['time_difference'].mean().reset_index().set_index('group')

            list_of_hit_rate.append(hit_rate)
            list_of_FA_rate.append(FA_rate)
            list_of_miss_rate.append(miss_rate)
            list_of_hit_latencies.append(hit_latencies)
            list_of_FA_latencies.append(FA_latencies)
            ID_list.append(parquet_file.split('.')[0])


    # Combine results from all files into a single DataFrame
    hit_rate_all = pd.concat(list_of_hit_rate, axis=1, ignore_index=True)
    FA_rate_all = pd.concat(list_of_FA_rate, axis=1,ignore_index=True)
    miss_rate_all = pd.concat(list_of_miss_rate, axis=1,ignore_index=True)
    hit_latencies_all = pd.concat(list_of_hit_latencies, axis=1, ignore_index=True)
    FA_latencies_all = pd.concat(list_of_FA_latencies, axis=1, ignore_index=True)

    # print(hit_rate_all)
    # print(hit_latencies_all)

    if flag ==1:

        # all_features = pd.concat([hit_rate_all, FA_rate_all, miss_rate_all, hit_latencies_all, FA_latencies_all], ignore_index=True)
        features = pd.concat([hit_rate_all, FA_rate_all, miss_rate_all, hit_latencies_all], ignore_index=True)
    
    elif flag==2:

        # hit, FA and miss rates only
        features = pd.concat([hit_rate_all, FA_rate_all, miss_rate_all], ignore_index=True)

    else:
        # hit rate only
        features = pd.concat([hit_rate_all], ignore_index=True)

        # print(all_features)

    return features, ID_list

num_clusters = 3

def plot_silhouette(df, clusters):

    import matplotlib.pyplot as plt

    silhouette_avg = silhouette_score(df, clusters)
    sample_silhouette_values = silhouette_samples(df, clusters)


    print(f"Silhouette Score: {silhouette_avg}")

    # Score per sample    
    y_lower = 10
    for j in range(num_clusters):
        
        ith_cluster_silhouette_values = sample_silhouette_values[clusters == j]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(j) / num_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))
        y_lower = y_upper + 10

    plt.title(f'Number of clusters: {num_clusters}\nSilhouette Score: {silhouette_avg:.2f}', fontsize=12)
    plt.xlabel('Silhouette Score')
    plt.ylabel('Cluster Label')

