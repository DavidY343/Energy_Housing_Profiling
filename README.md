# Energy Consumption Clustering Pipeline

Un pipeline completo de análisis y clustering de datos de consumo energético con interfaz web integrada. Este proyecto implementa múltiples algoritmos de agrupamiento (K-Means, Bisecting K-Means, Spectral Clustering) y técnicas de procesamiento de datos para identificar patrones de consumo en viviendas.

## Características Principales

- **Pipeline modular** con notebooks independientes para cada fase del análisis
- **Interfaz web** desarrollada en Flask para uso sin conocimientos técnicos
- **Múltiples algoritmos de clustering** con optimización automática de hiperparámetros
- **Generación automática de más de 40 métricas** estadísticas y espectrales
- **Visualizaciones interactivas** en 2D y 3D con PCA
- **Árboles de decisión** para interpretabilidad de clusters
- **Arquitectura extensible** para nuevos algoritmos y datasets

## Requisitos

- **Python 3.9+**
- Entorno virtual recomendado (`venv`, `conda` o similar)

## Instalación

1. Clona el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd [NOMBRE_DEL_PROYECTO]
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

Esto instalará automáticamente todas las librerías necesarias: pandas, scikit-learn, matplotlib, fancyimpute, Flask, entre otras.

## Estructura del Proyecto

```
├── Algorithm/                          # Núcleo del proyecto
│   ├── dataset/                       # Conjuntos de datos intermedios
│   ├── img_results/                   # Gráficos finales
│   ├── flowchart/                     # Diagramas de flujo
│   ├── pkls/                          # Modelos serializados
│   ├── data_analysis_results/         # Reportes de análisis
│   └── [notebooks 1-8].ipynb         # Pipeline principal
├── data_raw/                          # Datos originales (Datadis)
├── interfaz/                          # Interfaz web Flask
│   ├── src/                          # Módulos de procesamiento
│   └── app.py                        # Aplicación principal
├── DEPRECATED_USE_*/                  # Exploraciones anteriores
└── README.md
```

## Pipeline de Ejecución

### Opción 1: Notebooks (Uso Técnico)

Ejecuta los notebooks en el siguiente orden:

1. **`1_concatenate_csv_and_report.ipynb`**
   - Unifica archivos CSV individuales
   - Genera dataset consolidado y reporte inicial

2. **`2_EDA_and_preproccesed.ipynb`**
   - Análisis exploratorio de datos (EDA)
   - Tratamiento de valores atípicos y nulos

3. **`3_generate_features.ipynb`**
   - Calcula +40 métricas estadísticas y espectrales
   - Genera `dataset/generated_features.csv`

4. **`4_impute_new_data.ipynb`**
   - Imputación con 3 estrategias:
     - KNNImputer (vecinos cercanos)
     - SoftImpute (descomposición matricial)
     - Drop (eliminación conservadora)

5. **`5_scaler.ipynb`**
   - Estandarización con StandardScaler/MinMaxScaler

6. **`6_1_Select_K_Kmeans.ipynb`** / **`6_2_Select_K_BKmeans.ipynb`** / **`6_3_Select_K_Spectral_Clustering.ipynb`**
   - Selección óptima del número de clusters
   - Entrenamiento y serialización de modelos

7. **`7_1_visualization_kmeans_and_bkmeans.ipynb`** / **`7_2_visualization_spectral_clustering.ipynb`**
   - Visualizaciones 2D/3D con PCA
   - Gráficas de clusters vs features

8. **`8_binary_tree.ipynb`**
   - Árbol de decisión para interpretabilidad
   - Reglas de asignación a clusters

### Opción 2: Interfaz Web (Uso Simplificado)

Para usuarios sin conocimientos técnicos:

```bash
cd interfaz
python app.py
```

Accede a `http://127.0.0.1:5000/` y sigue el flujo guiado:

1. **Carga de Datos** - Selecciona carpeta con CSVs
2. **Generación e Imputación** - Cálculo automático de características
3. **Escalado y Clustering** - Selección de algoritmo y optimización
4. **Visualización** - Gráficos automáticos de resultados
5. **Interpretabilidad** - Árbol de decisión opcional

## Algoritmos Implementados

### Clustering
- **K-Means**: Clustering particional clásico
- **Bisecting K-Means**: Variante jerárquica divisiva
- **Spectral Clustering**: Basado en teoría de grafos

### Selección de K
- **Método K-ISAC_TLP**: Optimización automática de hiperparámetros
- **Métricas de validación**: Silhouette, Davies-Bouldin Index, MAE

### Imputación
- **KNN Imputer**: Basado en k-vecinos más cercanos
- **Soft Impute**: Descomposición matricial
- **Drop**: Eliminación conservadora


## Casos de Uso

- **Empresas energéticas**: Segmentación de clientes y tarifas personalizadas
- **Investigación académica**: Análisis de patrones de consumo
- **Políticas públicas**: Identificación de grupos vulnerables
- **Eficiencia energética**: Detección de consumos anómalos

## Módulos de la Interfaz

Los scripts en `interfaz/src/` encapsulan la lógica de cada fase:

- `EDA.py` - Análisis exploratorio
- `generate_features.py` - Cálculo de características
- `impute_data.py` - Imputación de datos
- `scaler.py` - Escalado de variables
- `select_k.py` - Optimización de clusters
- `visualization.py` - Generación de gráficos
- `binary_tree.py` - Interpretabilidad

## Visualizaciones

- **PCA 2D/3D**: Reducción dimensional para visualización
- **Centroides**: Perfiles promedio de cada cluster
- **Distribución**: Histogramas y boxplots por grupo
- **Mapas de calor**: Correlaciones entre variables
- **Árboles de decisión**: Reglas interpretables

## Formato de Datos

Los datos de entrada deben seguir el formato Datadis:

- Las columnas deben ser: `cups`, `fecha`, `hora`, `consumo_kWh`, `metodoObtencion`, `energiaVertida_kWh`.  

**Ejemplo de una fila de datos:**  
`0d9378df292f;2021/08/07;01:00;0,000;;`  

## Extensibilidad

El diseño modular permite fácilmente:

- **Nuevos algoritmos**: Añadir DBSCAN, GMM, etc.
- **Datasets diferentes**: Adaptar a nuevas ciudades/períodos
- **Métricas personalizadas**: Incorporar features específicas
- **Visualizaciones**: Crear nuevos tipos de gráficos


---

**Nota**: Este pipeline ha sido desarrollado como herramienta de investigación siguiendo buenas prácticas en ciencia de datos. La arquitectura modular facilita su adaptación a diferentes contextos y la incorporación de nuevas técnicas de análisis.