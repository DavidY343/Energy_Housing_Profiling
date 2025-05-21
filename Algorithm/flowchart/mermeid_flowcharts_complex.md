---
config:
  layout: dagre
---
flowchart TD
    %% --- EDA ---
    A["Concatenar CSVs"] --> A2["Análisis Individual por CSV"] 
    A2["Análisis Individual por CSV"] --> A3["Preprocesamiento de Datos Consolidado"]
    A3 --> A4["EDA Consolidado"]

    %% --- Generacion de Features ---
    A4 --> B["Generación de Features"]
    B --> B2{"Imputación de Datos"}
    B2 --> B3["KNN-Imputer"] & B5["Soft-Impute"]  & B4["Drop Missing Values"] 
    B3 --> B6{"Selección de Scaler"}
    B4 --> B6{"Selección de Scaler"}
    B5 --> B6{"Selección de Scaler"}
    B6 --> B7["StandarScaler"] & B8["Min-Max Scaler"] 
    B8 --> C{"Selección de K óptimo"}
    B7 --> C{"Selección de K óptimo"}
    C --> C1["Método del Codo"] & C2["Silhouette"] & C3["Davies-Bouldin"] & C4["K-ISAC_TLP"]
    
    %% --- MODELOS ---
    C1 --> D["Ejecución de Modelos"]
    C2 --> D
    C3 --> D
    C4 --> D
    D --> D1["K-means"] & D2["Bisecting K-means"] & D3["Spectral Clustering"] & D4["CKSC"]
    
    %% --- RESULTADOS ---
    D1 --> E{"Visualización"} & F["Explicabilidad"]
    D2 --> E{"Visualización"} & F["Explicabilidad"]
    D3 --> E{"Visualización"} & F["Explicabilidad"]
    D4 --> E{"Visualización"} & F["Explicabilidad"]
    
    F --> F2["Arbol de Decision"]
    F2 --> F3["Reglas de Decision"]
    E --> E2{"PCA"}
    E --> E5["Clusters vs Features"]
    E2 --> E3["2D"] & E4["3D"]
    E3 --> G["Interpretación"]
    E4 --> G["Interpretación"]
    E5 --> G["Interpretación"]
    F3 --> G

    style C stroke:#ff0000,stroke-width:3px
    style D stroke:#0000ff,stroke-width:3px
    style E stroke:#008000,stroke-width:3px
    style F stroke:#ffa500,stroke-width:3px
    style G stroke:#800080,stroke-width:3px