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
- **Features**:
  - Estadísticas básicas: `media`, `desviación estándar`, `máximo/mínimo`, `energía total`.
  - Avanzadas: `autocorrelación`, `coeficiente de variación`, `transformadas (Fourier/Wavelet)`, `núm. picos`.
  - Variables temporales: `día de la semana`, `hora pico`, `mes del año`, `estacion`.

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