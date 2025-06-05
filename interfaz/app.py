from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import pandas as pd
import numpy as np
from io import StringIO
from src.EDA import EDAEnergyConsumption
from src.generate_features import EnergyFeatureGenerator
from src.impute_data import MissingValueHandler
from src.scaler import FeatureScaler
from src.select_k import MyKMeans, MyBisectingKMeans, MySpectralClustering
from src.visualization import ClusterVisualizer, SpectralClusterVisualizer
from src.binary_tree import MyBinaryTree



app = Flask(__name__)
app.secret_key = 'agapito_secret_key'


current_script_dir = os.path.dirname(os.path.abspath(__file__))

def eda(data_raw_path=None, output_path=None):
    folder_path = os.path.join(current_script_dir, data_raw_path)
    output_dir = os.path.join(current_script_dir, output_path)
    merged_output_path = os.path.join(current_script_dir, 'dataset', 'vertical_data.csv')

    eda = EDAEnergyConsumption(
    folder_path=folder_path,
    output_dir=output_dir,
    merged_output_path=merged_output_path
    )

    eda.process_files()
    eda.explore_data_analysis(merged_output_path)

def generate_features():

    the_path = os.path.join(current_script_dir, 'dataset', 'vertical_data.csv')
    feature_generator = EnergyFeatureGenerator(the_path)

    feature_generator.pipeline_completo(False)

def impute_new_data():

    the_path = os.path.join(current_script_dir, 'dataset', 'generated_features.csv')
    imputer = MissingValueHandler(the_path)

    imputer.pipeline_completo()

def scale_data(which_imputer=None):

    if which_imputer == 'soft':
        the_path = os.path.join(current_script_dir, 'dataset', 'features_SoftImpute.csv')
    elif which_imputer == 'drop':
        the_path = os.path.join(current_script_dir, 'dataset', 'features_drop.csv')
    elif which_imputer == 'knn':
        the_path = os.path.join(current_script_dir, 'dataset', 'features_KNNImputer.csv')
    else:
        raise ValueError("El método de imputación debe ser 'soft', 'drop' o 'knn'.")
    scaler = FeatureScaler(the_path)

    scaler.pipeline_completo()


def select_k(which_model=None, which_scaler=None):

    if which_scaler == 'minmax':
        the_path = os.path.join(current_script_dir, 'dataset', 'features_MinMaxScaler.csv')
    elif which_scaler == 'standard':
        the_path = os.path.join(current_script_dir, 'dataset', 'features_StandardScaler.csv')
    else:
        raise ValueError("El escalador debe ser 'minmax' o 'standard'.")
    the_path = os.path.join(current_script_dir, 'dataset', 'features_StandardScaler.csv')
    if which_model == 'kmeans':
        kmeans_optimizer = MyKMeans(the_path)
        # Eligimos el rango de k entre 2 y 30 que fue el usado en el trabajo
        kmeans_optimizer.pipeline(range(2, 30))
    elif which_model == 'bkmeans':
        bkmeans_optimizer = MyBisectingKMeans(the_path)
        # Elegimos el rango de k entre 2 y 30 que fue el usado en el trabajo
        bkmeans_optimizer.pipeline(range(2, 30))
    elif which_model == 'sc':
        sc_optimizer = MySpectralClustering(the_path)
        sc_optimizer.pipeline(range(2, 15), gamma_values=[0.1, 0.5, 1.0, 2.0, 5.0], n_neighbors_values=[5, 10, 15, 20, 25])
    else:
        raise ValueError("El modelo debe ser 'kmeans', 'bkmeans' o 'sc'.")


def visualizer(which_model=None, which_scaler=None):
    if which_scaler == 'minmax':
        features_path = os.path.join(current_script_dir, 'dataset', 'features_MinMaxScaler.csv')
    elif which_scaler == 'standard':
        features_path = os.path.join(current_script_dir, 'dataset', 'features_StandardScaler.csv')
    else:
        raise ValueError("El escalador debe ser 'minmax' o 'standard'.")
    
    original_data_path = os.path.join(current_script_dir, 'dataset', 'vertical_data.csv')
    if which_model == 'kmeans':
        model_type = 'kmeans'
        visualizer = ClusterVisualizer(features_path=features_path, 
                                   original_data_path=original_data_path, 
                                   model_type=model_type)
        visualizer.pipeline()
    elif which_model == 'bkmeans':
        model_type = 'bkmeans'
        visualizer = ClusterVisualizer(features_path=features_path, 
                                   original_data_path=original_data_path, 
                                   model_type=model_type)
        visualizer.pipeline()
    elif which_model == 'sc':
        model_type = 'sc'
        visualizer = SpectralClusterVisualizer(features_path=features_path, 
                                               original_data_path=original_data_path)
        visualizer.pipeline()   
    else:
        raise ValueError("El modelo debe ser 'kmeans', 'bkmeans' o 'sc'.")

def binary_tree_player(which_model=None, which_scaler=None, which_imputer=None):
    if which_scaler == 'minmax':
        scaled_features_path = os.path.join(current_script_dir, 'dataset', 'features_MinMaxScaler.csv')
    elif which_scaler == 'standard':
        scaled_features_path = os.path.join(current_script_dir, 'dataset', 'features_StandardScaler.csv')
    else:
        raise ValueError("El escalador debe ser 'minmax' o 'standard'.")
    
    if which_imputer == 'soft':
        features_path = os.path.join(current_script_dir, 'dataset', 'features_SoftImpute.csv')
    elif which_imputer == 'drop':
        features_path = os.path.join(current_script_dir, 'dataset', 'features_drop.csv')
    elif which_imputer == 'knn':
        features_path = os.path.join(current_script_dir, 'dataset', 'features_KNNImputer.csv')
    else:
        raise ValueError("El método de imputación debe ser 'soft', 'drop' o 'knn'.")

    if which_model == 'kmeans':
        model_type = 'kmeans'
    elif which_model == 'bkmeans':
        model_type = 'bkmeans'
    elif which_model == 'sc':
        model_type = 'sc'
    else:
        raise ValueError("El modelo debe ser 'kmeans', 'bkmeans' o 'sc'.")

    binary_tree_interpreter = MyBinaryTree(features_path=features_path, 
                                           scaled_features_path=scaled_features_path, 
                                           model_type=model_type)
    binary_tree_interpreter.pipeline()



    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_EDA', methods=['POST'])
def run_EDA():
    try:
        data = request.json
        data_raw_path = data.get('dataRawPath')
        output_path = data.get('outputPath')
        
        # Validar parámetros
        if not all([data_raw_path, output_path]):
            return jsonify({'error': 'Faltan parámetros requeridos'}), 400
        
        # Hacer EDA
        eda(data_raw_path=data_raw_path, output_path=output_path)
        # Generar características
        generate_features()
        # Imputar datos
        impute_new_data()
        return jsonify({
            'status': 'success',
            'message': 'EDA completado exitosamente',
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/run_clustering', methods=['POST'])
def run_clustering():
    try:
        data = request.json
        imputation = data.get('imputation')
        scaling = data.get('scaling')
        model_type = data.get('model')
        
        # Nuevos parámetros para rutas
        data_raw_path = data.get('dataRawPath')
        output_path = data.get('outputPath')
        
        # Validar parámetros
        if not all([imputation, scaling, model_type, data_raw_path, output_path]):
            return jsonify({'error': 'Faltan parámetros requeridos'}), 400
        

        # Escalado de datos
        scale_data(which_imputer=imputation)
        
        # Seleccionar el modelo de clustering y ejecutar
        select_k(which_model=model_type, which_scaler=scaling)
        
        visualizations = {
            'mae': f'MAE_{model_type}.png',
            'silhouette': f'silhouette_{model_type}.png',
            'dbi': f'dbi_{model_type}.png',
            'irrelevant_clusters': f'irrelevant_clusters_{model_type}.png',
            'wcss': f'wcss_{model_type}.png',
        }
        return jsonify({
            'status': 'success',
            'message': 'Clustering completado exitosamente',
            'visualizations': visualizations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    try:
        data = request.json
        model_type = data.get('model')
        scaling = data.get('scaling')
        
        if not all([model_type, scaling]):
            return jsonify({'error': 'Faltan parámetros requeridos'}), 400
        
        # Visualizar
        visualizer(which_model=model_type, which_scaler=scaling) 
        
        # Generar visualizaciones
        visualizations = {
            'pca_2d': f'cluster_pca2d_{model_type}.png',
            'pca_3d': f'cluster_pca3d_{model_type}.png',
            'cluster_centers': f'cluster_centers_{model_type}.png',
            'cluster_distribution': f'cluster_distribution_{model_type}.png'
        }
        
        return jsonify({
            'status': 'success',
            'visualizations': visualizations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


    

@app.route('/binary_tree', methods=['GET', 'POST'])
def binary_tree():
    try:
        data = request.json
        model_type = data.get('model')
        scaling = data.get('scaling')
        impute = data.get('imputation')
        
        if not all([model_type, scaling, impute]):
            return jsonify({'error': 'Faltan parámetros requeridos'}), 400
        

        binary_tree_player(which_model=model_type, which_scaler=scaling, which_imputer=impute)
       
        # Generar visualizaciones
        visualizations = {
            'decision_tree': f'decision_tree_{model_type}.png',
            'feature_importance': f'feature_importance_{model_type}.png',
        }
        
        return jsonify({
            'status': 'success',
            'visualizations': visualizations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_image/<filename>')
def get_image(filename):
    try:
        # Asegúrate de que la ruta es segura
        safe_path = os.path.join(current_script_dir, 'img_results')
        return send_from_directory(safe_path, filename)
    except FileNotFoundError:
        return jsonify({'error': 'Image not found'}), 404
    
if __name__ == '__main__':
    app.run(debug=True)