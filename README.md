# TFG_electric_clustering

El id "c1244d6dea7a" tiene la hora con dos ceros de mas:
c1244d6dea7a;02/08/2023;24:00;0,001;;



Cosas que he ido descubriendo de los datos:

Las horas estaban a veces mal formateadas por ejemplo "1:00" en vez de "01:00" o "24:00:00" en vez de "24:00"
Existian dos formatos de fechas que he tenido que agrupar
Los datos de la serie temporal dependen de la vivienda algunso empiezan el 07-08-2021 y otros el 30-08-2021. Los he puesto todos a emepzar a partir del 30 para unificar
04666163609d Consumo 07-08-2021_07-08-2023 Este cups no acaba el 7, acaba el 5
03c8338d7f1d Consumo 07-08-2021_07-08-2023 Este cups no acaba el 7, acaba el 4
277f70c6024c Consumo 07-08-2021_07-08-2023 Este cups no empiza en el 7-08 empiza en el 12-05

Cosas que comentar a agapito:

Te comparto el github?
Que hago con los datos como planteo
Puedo hacer clustering en el formato que me pasó?
A dia de hoy mi idea del clustering va ser pasar todo a formato ancho y hacer clustering de eso (en todas las formas posibles: dendogramas, kmeans, DTW, SOM...)
Otra idea que he visto es solo hacer clustering por ejemplo de dias. Cojo los domingos y hago clustering de los domingos de todas las viviendas.

1. Preprocesamiento y clustering en formato ancho con tslearn
Aplanado de datos y formato ancho:
Transformar las series horarias en vectores de características (cada hora como una dimensión) es una buena idea para usar algoritmos como k-means. Sin embargo, ten en cuenta que el alto número de dimensiones puede afectar la distancia euclidiana.

Manejo de valores faltantes:
Es importante definir una estrategia robusta. Puedes probar imputaciones basadas en interpolación, imputación basada en vecinos o incluso métodos más avanzados que consideren la estacionalidad y tendencias. Si decides usar un marcador como '?' en algunos casos, asegúrate de que el algoritmo lo interprete o reemplázalo antes del clustering.

2. Experimentación con otros modelos de clustering
Modelos alternativos a k-means:
Algunas opciones interesantes son:

DTW (Dynamic Time Warping): Permite medir la similitud entre series que pueden estar desfasadas en el tiempo. Puedes usar k-medoids o clustering jerárquico con DTW.

SOM (Self-Organizing Maps): Es útil para visualizar y agrupar series con patrones complejos.

Clustering Espectral: Puede capturar estructuras no lineales en los datos.

K-ISAC_TLP: Si bien es menos común, puede aportar una perspectiva diferente en la detección de patrones.

Otros métodos: Considera también explorar algoritmos basados en modelos, como Gaussian Mixture Models, o técnicas de densidad (por ejemplo, DBSCAN adaptado para series temporales).

3. Reducción de dimensionalidad con PCA
Aplicar PCA:
Una vez que tengas tus series en formato ancho, aplicar PCA te ayudará a reducir la dimensionalidad y resaltar las componentes principales de variabilidad.

Comparativa de enfoques:
Repite el clustering (tanto con k-means como con otros métodos) en el espacio reducido. Esto puede ayudar a identificar si el “ruido” en los datos originales estaba ocultando patrones relevantes.

4. Incorporación de variables agregadas
Extracción de características relevantes:
Además de las variables que mencionas (día del mes, hora pico, consumo medio diario), podrías considerar:

Estadísticas de dispersión: Desviación estándar, coeficiente de variación.

Características de forma: Asimetría y curtosis, para identificar la distribución del consumo.

Medidas de estacionalidad: Por ejemplo, la diferencia entre consumo en días laborables y fines de semana, o variaciones durante diferentes momentos del día.

Transformadas en el dominio de la frecuencia: Usar la Transformada de Fourier para identificar ciclos o picos en la periodicidad.

Variables derivadas de tendencias: Como la tasa de cambio o el crecimiento acumulado en ciertos periodos.

Repetir el clustering:
Con este conjunto de variables, realiza nuevamente los análisis de clustering para ver si se capturan patrones que antes no eran tan evidentes.

5. Análisis en formato vertical
Formato vertical vs. ancho:
Organizar los datos en formato “long” (vertical) te permitirá trabajar con la serie en su forma original, preservando la secuencia temporal.

Ventajas potenciales:

Facilita el uso de técnicas de series temporales, como modelos autoregresivos o métodos de deep learning (por ejemplo, autoencoders LSTM) para extraer características.

Puede mejorar el manejo de series irregulares o con diferentes longitudes.

Consideraciones:
Es posible que necesites transformar o segmentar la serie para extraer ventanas de análisis comparables, pero esta aproximación puede revelar dinámicas temporales que se pierden en el formato ancho.

6. Análisis por subconjuntos y temporalidad
Segmentación por periodos:
Analizar solo determinados periodos (por ejemplo, meses de verano, fines de semana o horas específicas) te permitirá comprender si los patrones de consumo son consistentes o varían según el contexto.

Análisis contextual:
Además de segmentar por tiempo, podrías incorporar datos externos (como datos meteorológicos, festivos o indicadores socioeconómicos) para ver si influyen en los patrones de consumo.

Ideas adicionales y recomendaciones generales
Validación y estabilidad de clusters:
Sea cual sea el método que utilices, es fundamental evaluar la calidad de los clusters. Considera métricas como el índice de silueta, la consistencia interna de los clusters y, si es posible, validación con información externa (por ejemplo, clasificación basada en perfiles conocidos de consumidores).

Exploración de técnicas de Deep Learning:
Métodos como los autoencoders para series temporales pueden ayudar a aprender representaciones latentes que capturen la dinámica temporal, facilitando un clustering más robusto y menos sensible al ruido.

Visualización:
Usa técnicas de visualización (como t-SNE o UMAP) para proyectar los clusters en un espacio de 2D y poder interpretarlos de forma visual. Esto puede ayudarte a identificar outliers o patrones interesantes.

Iteración y comparación de métodos:
Dado que estás explorando varias técnicas (k-means, DTW, SOM, PCA, etc.), es recomendable diseñar un pipeline que te permita comparar de forma sistemática los resultados. Esto facilitará la identificación de qué enfoques ofrecen las agrupaciones más coherentes o interpretables para tu caso de estudio.

Documentación y justificación:
Asegúrate de documentar cada paso, justificar la elección de cada técnica y detallar las limitaciones de cada método. Esto no solo enriquecerá el contenido técnico de tu TFG, sino que también demostrará una comprensión profunda del problema y de las técnicas empleadas.