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
    B2 --> B6{"Selección de Scaler"}
    B6 --> C{"Selección de K óptimo"}
    
    %% --- MODELOS ---
    C --> D["Ejecución de Modelos"]
    
    %% --- RESULTADOS ---
    D --> E{"Visualización"} & F["Explicabilidad"]
    
    F --> F2["Arbol de Decision"]
    F2 --> F3["Reglas de Decision"]
    E --> E2{"PCA"}
    E --> E5["Clusters vs Features"]
    E2 --> G["Interpretación"]
    E5 --> G["Interpretación"]
    F3 --> G

    style C stroke:#ff0000,stroke-width:3px
    style D stroke:#0000ff,stroke-width:3px
    style E stroke:#008000,stroke-width:3px
    style F stroke:#ffa500,stroke-width:3px
    style G stroke:#800080,stroke-width:3px