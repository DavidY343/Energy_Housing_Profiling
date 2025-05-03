import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pickle

df = pd.read_csv("../../data/vertical_preprocessed_data.csv", sep=";")


## APLICACIÓN DE CRITERIOS DE LIMPIEZA ##
# Los mismos criterios de limpieza aplicados en el preprocesado de datos del paper.

# 1. Exclusión de consumidores no domésticos (consumo > 15 kWh en cualquier hora)
hourly_consumption_cols = [h for h in df['hora'].unique() if isinstance(h, (int, float))]
max_consumption_per_hour = 15  # kWh

df_pivot = df.pivot_table(index=['cups', 'fecha'], columns='hora', values='consumo_kWh')
print(f"✅ Registros iniciales: {len(df_pivot)}")

mask_non_domestic = (df_pivot > max_consumption_per_hour).any(axis=1)
df_filtered = df_pivot[~mask_non_domestic].copy()

# 2. Umbral mínimo de consumo diario (100 W = 2.4 kWh/día)
min_daily_consumption = 2.4  # kWh
daily_consumption = df_filtered.sum(axis=1)
mask_low_consumption = daily_consumption < min_daily_consumption
df_filtered = df_filtered[~mask_low_consumption]

print(f"✅ Registros eliminados por consumo diario bajo: {mask_low_consumption.sum()}")

# Rellenar NaN restantes (si los hay) con ffill + bfill por cups
df_filled = df_filtered.groupby('cups').ffill()
if df_filled.isna().sum().sum() > 0:
    df_filled = df_filled.groupby('cups').bfill()

print(f"\n✅ NaN después de rellenar: {df_filled.isna().sum().sum()}")
print(f"✅ Registros finales: {len(df_filled)}")

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


# Guardar todos los resultados
with open('pkls/k_values.pkl', 'wb') as f:
    pickle.dump(list(k_values), f)

with open('pkls/mae_values.pkl', 'wb') as f:
    pickle.dump(MAE_values, f)

with open('pkls/irreleCluster_values.pkl', 'wb') as f:
    pickle.dump(irreleCluster_values, f)