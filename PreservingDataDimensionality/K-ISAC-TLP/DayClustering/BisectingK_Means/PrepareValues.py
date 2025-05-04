import pandas as pd
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle

df_filled = pd.read_csv("../flatten_data_watts.csv", sep=",", index_col=[0, 1])

X = df_filled.values
k_values = range(2, 30)
MAE_values = []

# Umbral para considerar un cluster "irrelevante": 1% de los datos tal y como indica el paper
threshold_percent = 1
min_cluster_size = len(X) * threshold_percent / 100

irreleCluster_values = []

for k in k_values:
    bkm = BisectingKMeans(n_clusters=k, random_state=1)
    bkm.fit_predict(X)
    
    labels = bkm.labels_
    centroids = bkm.cluster_centers_

    mae = mean_absolute_error(X, centroids[labels])
    MAE_values.append(mae)

    unique_labels, counts = np.unique(labels, return_counts=True)
    irrele_clusters = sum(counts < min_cluster_size)
    irreleCluster_values.append(irrele_clusters)

with open('pkls/k_values.pkl', 'wb') as f:
    pickle.dump(list(k_values), f)

with open('pkls/mae_values.pkl', 'wb') as f:
    pickle.dump(MAE_values, f)

with open('pkls/irreleCluster_values.pkl', 'wb') as f:
    pickle.dump(irreleCluster_values, f)