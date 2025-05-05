# Estrategias de Clustering para Series Temporales ROADMAP

## 1. Mantener la dimensionalidad original

### 1.1. Métodos basados en formas de onda completas
- **K-Medoids con DTW** (usando `tslearn`):
  - *Problema*: Normalización min-max genera medoides alejados y Silhouette Score bajo (-1).
- **K-Means con DTW**:
  - *Limitación*: Requiere longitudes fijas (no aplicable directamente a series de distinta duración).

### 1.2. Clustering por días (estructura tabular)
- Tratar cada día como observación (`Y = vivienda y fecha`, `X = consumo 24H`).
- **Algoritmos**:
  - `K-Means` tradicional.
  - `Bisecting K-Means`.
  - **K-ISAC_TLP** (especializado en series temporales):
    - *Resultado*: Buen rendimiento (métricas como Silhouette/Elbow no fueron útiles aquí).

### 1.3. Subconjuntos temporales
Segmentar datos para analizar patrones específicos:
- Meses de **verano vs. invierno**.
- **Fines de semana vs. laborables**.
- Días concretos (ej. **domingos**).

### 1.4. Clustering basado en características
Extraer métricas globales y aplicar clustering sin alineación temporal:
#### 1.4.1. Estadísticas Básicas de Consumo por Día
- **`media`**: Consumo promedio del día (en Watts)
- **`std`**: Desviación estándar (variabilidad del consumo)
- **`max`** y **`min`**: Picos y valles de consumo
- **`total_energy`**: Suma total de energía consumida en el día (integral de la curva)
- **`range`**: Diferencia entre máximo y mínimo (amplitud de consumo)
- **`coef_var`**: Coeficiente de variación (`std/media`). Mide dispersión relativa (útil para comparar días con distintos niveles de consumo)

#### 1.4.2. Estadísticas Avanzadas de la Curva de Carga
- **`skewness`**: Asimetría de la distribución del consumo  
  *Ejemplo*: Si > 0, hay más días con consumo bajo pero con picos altos
- **`kurtosis`**: "Grosor" de las colas de la distribución  
  *Ejemplo*: Si > 3, la curva tiene picos más pronunciados que una normal
- **`peak_hour`**: Hora del día con máximo consumo (útil para detectar patrones horarios)
- **`num_peaks`**: Número de horas donde el consumo supera el percentil 90 (identifica días con múltiples picos)

#### 1.4.3. Ratios Temporales (Patrones de Consumo)
##### ratio_manana_tarde
$$
\text{Ratio} = \frac{\text{Consumo (8h-12h)}}{\text{Consumo (16h-20h)}}
$$
**Interpretación**: Si > 1, el consumo es mayor en la mañana que en la tarde

##### ratio_noche_dia
$$
\text{Ratio} = \frac{\text{Consumo (0h-6h)}}{\text{Consumo (7h-23h)}}
$$
**Interpretación**: Si alto, indica alto consumo nocturno (ej.: industrias que operan de noche)

#### 1.4.4. Autocorrelación (Patrones Temporales)
- **`autocorr_lag1`**: Correlación del consumo entre una hora y la siguiente (lag-1)  
  **Interpretación**:  
  - ≈ 1: Consumo muy predecible hora a hora (patrón estable)  
  - ≈ 0: Comportamiento errático

#### 1.4.5. Variables Temporales Externas
- **`dia_semana`**: Número del día (0=Lunes, ..., 6=Domingo)
- **`es_fin_semana`**: Binario (1 si es sábado o domingo)
- **`estacion`**: Estación del año (1=Invierno, 2=Primavera, etc.)

## 2. Reducción de dimensionalidad o filtrado de datos  
**Opción a valorar (si otras alternativas fallan):**  
- **Filtrado**: Eliminar viviendas/cups con insuficientes datos y ajustar a la máxima longitud común para homegeneizar.  
- **Enfoques propuestos**:  
  - `K-Means + DTW` (usando `tslearn`).  
  - `K-ISAC_TLP` (especializado en series temporales).  
  - `Clustering espectral`.  
- **Alternativa con PCA**:  
  - `SOM` (Mapas auto-organizados).  
  - `K-Means` tradicional.  
  - `Clustering jerárquico`.  
  - `Clustering espectral`.  

## 3. Otras ideas exploratorias  
- **Autoencoders para series temporales**:  
  - Compresión no lineal de dimensionalidad antes de clustering.  
- **Dendogramas**:
  - Complejidad computacional alta `O(n³) o O(n² log n)`(para métodos optimizados como linkage). Para miles de series temporales, puede volverse lento o inviable.