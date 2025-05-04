import pandas as pd
import warnings
warnings.filterwarnings("ignore")

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