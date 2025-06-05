import pandas as pd
import os
from datetime import datetime, timedelta
from textwrap import dedent
import re
from collections import defaultdict
from matplotlib import pyplot as plt

from src.EDA import EDAEnergyConsumption
from src.generate_features import EnergyFeatureGenerator
from src.impute_data import MissingValueHandler
from src.scaler import FeatureScaler
from src.select_k import MyKMeans, MyBisectingKMeans, MySpectralClustering
from src.visualization import ClusterVisualizer, SpectralClusterVisualizer
from src.unify import UnifyClusters
from src.binary_tree import MyBinaryTree

current_script_dir = os.path.dirname(os.path.abspath(__file__))
def eda():
    folder_path = os.path.join(current_script_dir, '..', 'data_raw')
    output_dir = os.path.join(current_script_dir, 'data_analysis_results')
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

def scale_data():

    # Usamos KNNImputer por ser la mejor opción, pero se puede cambiar a SoftImpute si se desea o drop filas con NaN
    the_path = os.path.join(current_script_dir, 'dataset', 'features_KNNImputer.csv')
    scaler = FeatureScaler(the_path)

    scaler.pipeline_completo()


def select_k(which_model=None):

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
        # Elegimos el rango de k entre 2 y 30 que fue el usado en el trabajo
        sc_optimizer.pipeline(range(2, 15), gamma_values=[0.1, 0.5, 1.0, 2.0, 5.0], n_neighbors_values=[5, 10, 15, 20, 25])
    else:
        raise ValueError("El modelo debe ser 'kmeans', 'bkmeans' o 'sc'.")


def visualize(which_model=None):
    features_path = os.path.join(current_script_dir, 'dataset', 'features_StandardScaler.csv')
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

def binary_tree(which_model=None):
    features_path = os.path.join(current_script_dir, 'dataset', 'features_KNNImputer.csv')
    scaled_features_path = os.path.join(current_script_dir, 'dataset', 'features_StandardScaler.csv')

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

def unify_clusters():
    """Unifica los clusters generados por KMeans, Bisecting KMeans y Spectral Clustering en un único archivo CSV."""
    kmeans_path = os.path.join(current_script_dir, 'dataset', 'cluster_cups_kmeans.csv')
    bkmeans_path = os.path.join(current_script_dir, 'dataset', 'cluster_cups_bkmeans.csv')
    sc_path = os.path.join(current_script_dir, 'dataset', 'cluster_cups_sc.csv')
    output_path = os.path.join(current_script_dir, 'dataset', 'clusters_unificado.csv')

    unify = UnifyClusters(kmeans_path=kmeans_path, 
                          bkmeans_path=bkmeans_path, 
                          sc_path=sc_path, 
                          output_path=output_path)
    unify.unify()

def main():
    #eda()
    #generate_features()
    #impute_new_data()
    #scale_data()
    select_k('kmeans')
    visualize('kmeans')
    binary_tree('kmeans')
    unify_clusters()


    
if __name__ == "__main__":
    main()