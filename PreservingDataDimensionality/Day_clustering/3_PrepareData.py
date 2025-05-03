import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle

df = pd.read_csv("../../data/vertical_preprocessed_data.csv", sep=";")

# Pivotar la tabla para tener un perfil de consumo por día
df_pivot = df.pivot_table(index=['cups', 'fecha'], columns='hora', values='consumo_kWh')

# Dado que existen NaN, procederemos a rellenarlos con el método de forward fill (ffill).
# Principalmente, ocurren a las 2 am y 3 am, donde no hay consumo registrado.

df_filled = df_pivot.groupby('cups').ffill()  

if df_filled.isna().sum().sum() > 0:
    df_filled = df_filled.groupby('cups').bfill()

print(f"\n✅ NaN después de rellenar: {df_filled.isna().sum().sum()}")

df_filled = df_filled * 1000

df_filled.to_csv('flatten_data_watts.csv')

X = df_filled.values
k_values = range(2, 30)
MAE_values = []
# Umbral para considerar un cluster "irrelevante": 1% de los datos tal y como indica el paper
threshold_percent = 1
min_cluster_size = len(X) * threshold_percent / 100

irreleCluster_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
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