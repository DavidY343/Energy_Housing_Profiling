import pandas as pd
import warnings
from scipy import stats
import numpy as np

warnings.filterwarnings("ignore")

df = pd.read_csv("../../../data/vertical_preprocessed_data.csv", sep=";")

# Pivotar la tabla para tener un perfil de consumo por día
df_pivot = df.pivot_table(index=['cups', 'fecha'], columns='hora', values='consumo_kWh')

# Dado que existen NaN, procederemos a rellenarlos con el método de forward fill (ffill).
# Principalmente, ocurren a las 2 am y 3 am, donde no hay consumo registrado.

df_filled = df_pivot.groupby('cups').ffill()  

if df_filled.isna().sum().sum() > 0:
    df_filled = df_filled.groupby('cups').bfill()

print(f"\n NaN después de rellenar: {df_filled.isna().sum().sum()}")

df_filled = df_filled * 1000

# ---- CALCULAR VARIABLES AGREGADAS ----
def calculate_features(df):
    zero_division_count = {'coef_var': 0, 'ratio_manana_tarde': 0}
    constant_data_count = {'skewness': 0, 'kurtosis': 0}
    
    def safe_divide(a, b, feature_name):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            zero_mask = (b == 0)
            zero_count = zero_mask.sum()
            if zero_count > 0:
                zero_division_count[feature_name] += zero_count
        return result
    

    mean_vals = df.mean(axis=1)
    std_vals = df.std(axis=1)
    max_vals = df.max(axis=1)
    min_vals = df.min(axis=1)
    
    df_features = pd.DataFrame({
        'media': mean_vals,
        'std': std_vals,
        'max': max_vals,
        'min': min_vals,
        'range': max_vals - min_vals,
        'coef_var': safe_divide(std_vals, mean_vals, 'coef_var'),
    })
    
    def safe_skew(x):
        if x.nunique() <= 1:
            constant_data_count['skewness'] += 1
            return 0
        return stats.skew(x)
    
    def safe_kurtosis(x):
        if x.nunique() <= 1:
            constant_data_count['kurtosis'] += 1
            return 0
        return stats.kurtosis(x)
    
    df_features['skewness'] = df.apply(safe_skew, axis=1)
    df_features['kurtosis'] = df.apply(safe_kurtosis, axis=1)

    df_features['peak_hour'] = df.idxmax(axis=1)
    df_features['num_peaks'] = df.apply(lambda x: sum(x > x.quantile(0.9)), axis=1)
    
    df_features['ratio_manana_tarde'] = safe_divide(
        df.loc[:, 7:14].sum(axis=1),
        df.loc[:, 16:22].sum(axis=1),
        'ratio_manana_tarde'
    )
    
    print("\n=== RESUMEN DE ADVERTENCIAS ===")
    print(f"Divisiones por cero:")
    for k, v in zero_division_count.items():
        print(f"- {k}: {v} casos")
    
    print(f"\nDatos constantes o insuficientes:")
    for k, v in constant_data_count.items():
        print(f"- {k}: {v} casos")
    
    total_warnings = sum(zero_division_count.values()) + sum(constant_data_count.values())
    print(f"\nTOTAL ADVERTENCIAS: {total_warnings} casos problemáticos manejados")
    
    return df_features

df_features = calculate_features(df_filled)

df_features.reset_index(inplace=True)
df_features['fecha'] = pd.to_datetime(df_features['fecha'])
df_features['dia_semana'] = df_features['fecha'].dt.dayofweek  # 0=Lunes 1=Martes, ..., 6=Domingo
df_features['es_fin_semana'] = df_features['dia_semana'].isin([5,6]).astype(int)
df_features['estacion'] = (df_features['fecha'].dt.month % 12 + 3) // 3  # 1=Invierno, 2=Primavera, etc.

final_features_df = df_features.drop(columns=[col for col in df_features.columns if isinstance(col, (int, float)) and 0 <= col <= 23], errors='ignore')

print(f"\nNaN en el dataset de características: {final_features_df.isna().sum().sum()}")

## Escalar las características¿?
## Preguntar profesor

final_features_df.to_csv('aggregated_features_only.csv', index=False)