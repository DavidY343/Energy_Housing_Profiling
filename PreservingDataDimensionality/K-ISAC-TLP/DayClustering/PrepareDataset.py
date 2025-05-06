import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("../../../data/vertical_preprocessed_data.csv", sep=";")

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