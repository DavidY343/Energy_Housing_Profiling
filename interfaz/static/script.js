// Función para seleccionar opciones
function selectOption(element, type)
{
    const buttons = document.querySelectorAll(`.btn-option[onclick*="${type}"]`);
    buttons.forEach(btn => btn.classList.remove('active', 'active-option'));
    
    element.classList.add('active', 'active-option');
    
    const inputId = `selected${type.charAt(0).toUpperCase() + type.slice(1)}`;
    document.getElementById(inputId).value = element.dataset.value;
    
    updateActionButtons();
}

// Función para actualizar botones de acción (Visualizar/Árbol)
function updateActionButtons()
{
    const modelSelected = document.getElementById('selectedModel').value;
    const scalingSelected = document.getElementById('selectedScaling').value;
    const selectedImputation = document.getElementById('selectedImputation').value;
    const dataPath = document.getElementById('dataPath').value;
    const outputPath = document.getElementById('outputPath').value;
    
    const btnVisualize = document.getElementById('btnVisualize');
    const btnBinaryTree = document.getElementById('btnBinaryTree');
    const btnRunEDA = document.getElementById('btnRunEDA');
    const btnRunClustering = document.getElementById('btnRunClustering');

    // Habilita "EDA" solo si hay ruta de datos y salida
    btnRunEDA.disabled = !(dataPath && outputPath);

    // Habilita "Clustering" solo si hay modelo, scaling e imputacion seleccionado
    btnRunClustering.disabled = !(modelSelected && scalingSelected && selectedImputation);

    // Habilita "Visualizar" solo si hay modelo y escalado seleccionados
    btnVisualize.disabled = !(modelSelected && scalingSelected);
    
    // Habilita "Árbol Binario" solo si hay modelo, scaling e imputacion seleccionado 
    btnBinaryTree.disabled = !(modelSelected && scalingSelected && selectedImputation);
}
        
// Función para ejecutar el proceso de EDA
function runEDA()
{
    const config = {
        dataRawPath: document.getElementById('dataPath').value,
        outputPath: document.getElementById('outputPath').value
    };
    
    alert('Iniciando el proceso de extraccion de datos. Esto puede tardar aproximadamente 20 segundos, por favor espere...');

    fetch('/run_EDA', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            alert(`${data.message}\n`);
        }
    });
}

// Función para ejecutar el proceso de clustering
function runClustering()
{
    const config = {
        imputation: document.getElementById('selectedImputation').value,
        scaling: document.getElementById('selectedScaling').value,
        model: document.getElementById('selectedModel').value,
        dataRawPath: document.getElementById('dataPath').value,
        outputPath: document.getElementById('outputPath').value
    };
    
    alert('Iniciando el proceso de clustering. Esto puede tardar aproximadamente 10 segundos, por favor espere...');

    fetch('/run_clustering', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            alert(`${data.message}\n`);
            displayMetrics(data.visualizations);
        }
    });
}
// Función para mostrar visualizaciones
function displayMetrics(visualizations)
{
    const container = document.getElementById('visualization-content');
    container.innerHTML = '';
    
    // Crear pestañas para cada visualización
    container.innerHTML = `
    <ul class="nav nav-tabs" id="vizTabs" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#mae" type="button">
                MAE
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#sil" type="button">
                SILHOUETTE
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#dbi" type="button">
                Davis Bouldin Index
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#wcss" type="button">
                ELBOW METHOD
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#irrele" type="button">
                IRRELEVANT CLUSTERS
            </button>
        </li>
    </ul>
    <div class="tab-content p-3 border border-top-0 rounded-bottom">
        <div class="tab-pane fade show active" id="mae" role="tabpanel">
            <img src="/get_image/${visualizations.mae}" class="img-fluid">
        </div>
        <div class="tab-pane fade" id="sil" role="tabpanel">
            <img src="/get_image/${visualizations.silhouette}" class="img-fluid">
        </div>
        <div class="tab-pane fade" id="dbi" role="tabpanel">
            <img src="/get_image/${visualizations.dbi}" class="img-fluid">
        </div>
        <div class="tab-pane fade" id="wcss" role="tabpanel">
            <img src="/get_image/${visualizations.wcss}" class="img-fluid">
        </div>
        <div class="tab-pane fade" id="irrele" role="tabpanel">
            <img src="/get_image/${visualizations.irrelevant_clusters}" class="img-fluid">
        </div>
    </div>
    `;
    
    const tooltipTriggerList = [].slice.call(container.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Visualizar resultados
document.getElementById('btnVisualize').addEventListener('click', function()
{
    const config = {
        imputation: document.getElementById('selectedImputation').value,
        scaling: document.getElementById('selectedScaling').value,
        model: document.getElementById('selectedModel').value,
        dataRawPath: document.getElementById('dataPath').value, 
        outputPath: document.getElementById('outputPath').value
    };

    if (!selectedModel || !selectedScaling) {
    alert('⚠️ Por favor, selecciona un modelo y método de escalado primero');
    return;
    }

    alert('Iniciando el proceso de visualizacion. Esto puede tardar aproximadamente 10 segundos, por favor espere...');

    fetch('/visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            alert('Visualizaciones generadas correctamente.');
            displayVisualizations(data.visualizations);
        }
    });
});
        
// Función para mostrar visualizaciones de clustering
function displayVisualizations(visualizations)
{
    const container = document.getElementById('visualization-content');
    container.innerHTML = '';
    
    // Crear pestañas para cada visualización
    container.innerHTML = `
    <ul class="nav nav-tabs" id="vizTabs" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#pca2d" type="button">
                PCA 2D
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#pca3d" type="button">
                PCA 3D
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#centers" type="button">
                Centros
            </button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#distribution" type="button">
                Distribución
            </button>
        </li>
    </ul>
    <div class="tab-content p-3 border border-top-0 rounded-bottom">
        <div class="tab-pane fade show active" id="pca2d" role="tabpanel">
            <img src="/get_image/${visualizations.pca_2d}" class="img-fluid">
        </div>
        <div class="tab-pane fade" id="pca3d" role="tabpanel">
            <img src="/get_image/${visualizations.pca_3d}" class="img-fluid">
        </div>
        <div class="tab-pane fade" id="centers" role="tabpanel">
            <img src="/get_image/${visualizations.cluster_centers}" class="img-fluid">
        </div>
        <div class="tab-pane fade" id="distribution" role="tabpanel">
            <img src="/get_image/${visualizations.cluster_distribution}" class="img-fluid">
        </div>
    </div>
    `;
    
    const tooltipTriggerList = [].slice.call(container.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}


// Evento para el botón de clustering
document.getElementById('btnBinaryTree').addEventListener('click', function()
{
    const config = {
        imputation: document.getElementById('selectedImputation').value,
        scaling: document.getElementById('selectedScaling').value,
        model: document.getElementById('selectedModel').value,
        dataRawPath: document.getElementById('dataPath').value, 
        outputPath: document.getElementById('outputPath').value
    };

    if (!selectedModel || !selectedScaling || !selectedImputation) {
        alert('Por favor selecciona un modelo, un scaler y un imputer primero');
        return;
    }
    alert('Iniciando el proceso de visualizacion. Esto puede tardar aproximadamente 5 segundos, por favor espere...');

    fetch('/binary_tree', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            alert('Arbol generado correctamente.');
            displayBinaryTreeResults(data.visualizations);
        }
    })
    .catch(error => console.error('Error:', error));
});

// Función para mostrar resultados
function displayBinaryTreeResults(visualizations)
{
    const container = document.getElementById('visualization-content');
    
    container.innerHTML = `
    <div class="row">
        <div class="col-md-6 mb-4">
            <div class="card h-100">
                <div class="card-header bg-success text-white">
                    Árbol de Decisión
                </div>
                <div class="card-body">
                    <img src="/get_image/${visualizations.decision_tree}?t=${Date.now()}" 
                         class="img-fluid" 
                         alt="Árbol de decisión"
                         onerror="this.onerror=null;this.src='/static/fallback.png'">
                </div>
            </div>
        </div>
        <div class="col-md-6 mb-4">
            <div class="card h-100">
                <div class="card-header bg-success text-white">
                    Importancia de Características
                </div>
                <div class="card-body">
                    <img src="/get_image/${visualizations.feature_importance}?t=${Date.now()}" 
                         class="img-fluid" 
                         alt="Importancia de características"
                         onerror="this.onerror=null;this.src='/static/fallback.png'">
                </div>
            </div>
        </div>
    </div>
    `;
}