import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, BisectingKMeans, SpectralClustering
from sklearn.metrics import silhouette_score, mean_absolute_error, davies_bouldin_score
import math
import random

import pickle
import os

class K_ISAC_TLP:
    def __init__(self):
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def random_color(self):
        x = [0, 1, 2]
        y = [0, 1, 2]
        r = random.random()
        b = random.random()
        g = random.random()
        color_ = (r, g, b)
        return color_

    def calculateArea(self, p1,p2,p3):
        areaTriangle = abs (1/2 * ( ( (p2[0]*p1[1])-(p1[0]*p2[1]) )+ ( (p3[0]*p2[1])-(p2[0]*p3[1]) ) +
        ( (p1[0]*p3[1])-(p3[0]*p1[1]) ) ) )
        return areaTriangle

    def calculateSlope(self, p1,p2):
        slopeTwoPoints = math.degrees(math.atan((p2[1]-p1[1])/(p2[0]-p1[0])))
        return slopeTwoPoints

    def RenderTriangle(self, p1, p2, p3, ax, facecolor, edgecolor="none", linewidth=1):
        p = np.array([p1,p2,p3])
        triangle = plt.Polygon(p, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=0.7)
        ax.add_patch(triangle) 

    # --- Función ISAC (adaptada para una sola curva) ---
    def ISAC(self, k_values, measure_values, distanceBetweenPoints=3, consecutStability=3, 
            areaThreshold=None, slopeThreshold=None, curve_type="MAE", ax=None):
        """
        Parámetros:
            - k_values: Lista de valores de k (ej: range(2, 21)).
            - measure_values: Valores de la métrica (MAE o clusters irrelevantes).
            - distanceBetweenPoints: Distancia entre puntos del triángulo (default=3).
            - consecutStability: Número de triángulos consecutivos estables (default=3).
            - areaThreshold: Umbral de área (si None, se calcula para MAE).
            - slopeThreshold: Umbral de pendiente (si None, se calcula para MAE).
            - curve_type: "MAE" o "IRRELE" (para clusters irrelevantes).
            - ax: Eje matplotlib para graficar (opcional).
        """
        kMin = min(k_values)
        kMax = max(k_values)
        measureArray = measure_values

        # Calcular umbrales adaptativos si es MAE
        if curve_type == "MAE" and (areaThreshold is None or slopeThreshold is None):
            p1 = [kMin, measureArray[0]]
            p3 = [kMax, measureArray[-1]]
            p2 = [(kMin + kMax) // 2, measureArray[len(measureArray) // 2]]
            
            # Triángulo pequeño centrado
            p2_less = p2
            p1_less = [p2[0] - distanceBetweenPoints, p2[1] + (p1[1] - p2[1]) * (distanceBetweenPoints / (p2[0] - kMin))]
            p3_less = [p2[0] + distanceBetweenPoints, p2[1] + (p3[1] - p2[1]) * (distanceBetweenPoints / (kMax - p2[0]))]
            
            areaThreshold = self.calculateArea(p1_less, p2_less, p3_less)
            slopeThreshold = self.calculateSlope(p1, p3)

        # Umbrales fijos para clusters irrelevantes
        elif curve_type == "IRRELE":
            if areaThreshold is None:
                areaThreshold = 1.5  # Valor fijo igual que en el paper
            if slopeThreshold is None:
                slopeThreshold = 22.5  # Grados fijo igual que en el paper

        # --- Lógica de triangulación ---
        AREAS = []
        SLOPES = []
        positionsStopAreas = []
        positionsStopSlopes = []

        for k in range(0, len(measureArray) - (distanceBetweenPoints * 2)):
            p1 = [k + kMin, measureArray[k]]
            p2 = [k + kMin + distanceBetweenPoints, measureArray[k + distanceBetweenPoints]]
            p3 = [k + kMin + (distanceBetweenPoints * 2), measureArray[k + (distanceBetweenPoints * 2)]]

            area = self.calculateArea(p1, p2, p3)
            slope = self.calculateSlope(p1, p3)

            AREAS.append(area)
            SLOPES.append(slope)

            # Criterio de área
            if area <= areaThreshold:
                positionsStopAreas.append(k + kMin)
            # Criterio de pendiente
            if slope >= slopeThreshold:
                positionsStopSlopes.append(k + kMin)

        # Encontrar k óptimos (intersección de criterios)
        common_k_values = list(set(positionsStopAreas) & set(positionsStopSlopes))
        best_k = min(common_k_values) if common_k_values else min(positionsStopAreas)

        # --- Visualización (opcional) ---
        if ax is not None:
            ax.plot(k_values, measureArray, marker="o", color="red", markersize=2, linewidth=1)
            ax.set_xlabel("k")
            ax.set_ylabel("MAE" if curve_type == "MAE" else "Clusters irrelevantes")
            ax.set_title(f"Curva de {curve_type}")

            # Resaltar triángulos válidos
            if best_k is not None:
                num_triangles = 3
                colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(num_triangles)]

                for i, offset in enumerate(range(num_triangles)):
                    p1 = [best_k + offset, measureArray[best_k - kMin + offset]]
                    p2 = [best_k + offset + distanceBetweenPoints, measureArray[best_k - kMin + offset + distanceBetweenPoints]]
                    p3 = [best_k + offset + (distanceBetweenPoints * 2), measureArray[best_k - kMin + offset + (distanceBetweenPoints * 2)]]
                    self.RenderTriangle(p1, p2, p3, ax, facecolor=colors[i], edgecolor="black", linewidth=2)

        return {
        "AREAS": AREAS,
        "SLOPES": SLOPES,
        "k_area_true": positionsStopAreas,
        "k_slope_true": positionsStopSlopes,
        "k_common": common_k_values,
        "best_k": best_k
        }
    
    def k_ISAC_TLP(self, k_values, MAE_values, irreleCluster_values, distanceBetweenPoints=3, consecutStability=3, save_path=None):
        """
        Muestra resultados detallados para ambas curvas y combina los k óptimos.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        results_mae = self.ISAC(
            k_values, MAE_values, distanceBetweenPoints, consecutStability,
            areaThreshold=None, slopeThreshold=None, curve_type="MAE", ax=ax1
        )

        results_irrele = self.ISAC(
            k_values, irreleCluster_values, distanceBetweenPoints, consecutStability,
            areaThreshold=1.5, slopeThreshold=22.5, curve_type="IRRELE", ax=ax2
        )

        print("\n--- Resultados para MAE ---")
        print("Áreas de triángulos:", [round(a, 2) for a in results_mae["AREAS"]])
        print("Pendientes de triángulos:", [round(s, 2) for s in results_mae["SLOPES"]])
        print("K values donde área es TRUE:", results_mae["k_area_true"])
        print("K values donde pendiente es TRUE:", results_mae["k_slope_true"])
        print("K values comunes:", results_mae["k_common"])
        print("Mejor k (MAE):", results_mae["best_k"])

        print("\n--- Resultados para Clusters Irrelevantes ---")
        print("Áreas de triángulos:", [round(a, 2) for a in results_irrele["AREAS"]])
        print("Pendientes de triángulos:", [round(s, 2) for s in results_irrele["SLOPES"]])
        print("K values donde área es TRUE:", results_irrele["k_area_true"])
        print("K values donde pendiente es TRUE:", results_irrele["k_slope_true"])
        print("K values comunes:", results_irrele["k_common"])
        print("Mejor k (Clusters irrelevantes):", results_irrele["best_k"])

        print("\n--- Resultado Final ---")
        k_common_mae = set(results_mae["k_common"])
        k_common_irrele = set(results_irrele["k_common"])

        interseccion_comun = sorted(k_common_mae & k_common_irrele)

        if interseccion_comun:
            best_k = interseccion_comun[0]
            print(f"Mejor k común en ambas curvas: {best_k}")
        else:
            print("No hay valores comunes entre ambos criterios. Se tomarán decisiones individuales.")
            best_k = min(results_mae["best_k"], results_irrele["best_k"])

        plt.tight_layout()
        #plt.show()
        if save_path:
            fig.savefig(save_path)
        return best_k
    
class MyBisectingKMeans:
    def __init__(self, data_path="dataset/features_StandardScaler.csv"):
        """Inicializa con los datos escalados"""

        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_csv(data_path, index_col='cups')
        self.X = self.df.values
        self.bkmeans_model = None
    
    def find_optimal_k(self, k_range=range(2, 30), threshold_percent=5):
        """
        Encuentra el k óptimo evaluando múltiples métricas
        
        Args:
            k_range: Rango de valores k a evaluar
            threshold_percent: Porcentaje para considerar clusters irrelevantes
            
        Returns:
            Diccionario con resultados de las métricas
        """
        results = {
            'k_values': list(k_range),
            'MAE': [],
            'silhouette': [],
            'dbi': [],
            'irrelevant_clusters': [],
            'wcss': []
        }
        
        min_cluster_size = len(self.X) * threshold_percent / 100
        
        for k in k_range:
            bkmeans = BisectingKMeans(n_clusters=k, random_state=42).fit(self.X)
            labels = bkmeans.labels_
            
            # Calcular métricas
            results['MAE'].append(mean_absolute_error(self.X, bkmeans.cluster_centers_[labels]))
            results['silhouette'].append(silhouette_score(self.X, labels))
            results['dbi'].append(davies_bouldin_score(self.X, labels))
            results['wcss'].append(bkmeans.inertia_)
            
            # Contar clusters irrelevantes
            counts = np.unique(labels, return_counts=True)[1]
            results['irrelevant_clusters'].append(sum(counts < min_cluster_size))
        
        return results
    
    def plot_metrics(self, results):
        """Grafica las métricas para selección de k"""
        metrics = {
        'MAE': {'color': 'bo-', 'title': 'MAE vs k'},
        'silhouette': {'color': 'go-', 'title': 'Silhouette Score vs k'},
        'dbi': {'color': 'ro-', 'title': 'Davies-Bouldin Index vs k'},
        'irrelevant_clusters': {'color': 'mo-', 'title': 'Clusters Irrelevantes vs k'},
        'wcss': {'color': 'co-', 'title': 'WCSS vs k'}
        }
        save_paths = {
            'MAE': os.path.join(self.current_script_dir, '..', 'img_results', 'MAE_bkmeans.png'),
            'silhouette': os.path.join(self.current_script_dir, '..', 'img_results', 'silhouette_bkmeans.png'),
            'dbi': os.path.join(self.current_script_dir, '..', 'img_results', 'dbi_bkmeans.png'),
            'irrelevant_clusters': os.path.join(self.current_script_dir, '..', 'img_results', 'irrelevant_clusters_bkmeans.png'),
            'wcss': os.path.join(self.current_script_dir, '..', 'img_results', 'wcss_bkmeans.png')
        }
        for metric, config in metrics.items():
            plt.figure(figsize=(10, 5))
            plt.plot(results['k_values'], results[metric], config['color'], linewidth=2, markersize=8, markerfacecolor='white')
            plt.title(config['title'], pad=15, fontsize=14)
            plt.xlabel('Number of clusters (k)', labelpad=10, fontsize=12)
            plt.ylabel(metric, labelpad=10, fontsize=12)
            plt.xticks(results['k_values'][::2])
            plt.grid(True, linestyle='--', alpha=0.7)

            if save_paths and metric in save_paths:
                plt.savefig(save_paths[metric])
            plt.close()
    
    def train_final_model(self, k, save_path=None):
        """
        Entrena el modelo final con el k seleccionado
        
        Args:
            k: Número de clusters seleccionado
            save_path: Ruta para guardar el modelo
            
        Returns:
            Modelo bKMeans entrenado
        """
        self.bkmeans_model = BisectingKMeans(n_clusters=k, random_state=42).fit(self.X)
        
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(self.bkmeans_model, f)
        


    def pipeline(self, k_range=range(2, 30)):
        """
        Pipeline completo simplificado
        
        Args:
            k_range: Rango de valores k a evaluar
        """
        print("Evaluando métricas para diferentes valores de k...")
        results = self.find_optimal_k(k_range)

        self.plot_metrics(results)
        
        k_isac_tlp = K_ISAC_TLP()
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', 'k-isac_tlp_bkmeans.png')
        final_k = k_isac_tlp.k_ISAC_TLP(
            results['k_values'], 
            results['MAE'], 
            results['irrelevant_clusters'],
            save_path=save_path
        )
        print(f"\nK óptimo sugerido: {final_k} (basado en K-ISAC TLP)")
        
        print("\nEntrenando modelo final...")
        save_path = os.path.join(self.current_script_dir, '..', 'pkls', 'bkmeans_model.pkl')
        self.train_final_model(final_k, save_path=save_path)


class MyKMeans:
    def __init__(self, data_path="dataset/features_StandardScaler.csv"):
        """Inicializa con los datos escalados"""

        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_csv(data_path, index_col='cups')
        self.X = self.df.values
        self.kmeans_model = None
    
    def find_optimal_k(self, k_range=range(2, 30), threshold_percent=5):
        """
        Encuentra el k óptimo evaluando múltiples métricas
        
        Args:
            k_range: Rango de valores k a evaluar
            threshold_percent: Porcentaje para considerar clusters irrelevantes
            
        Returns:
            Diccionario con resultados de las métricas
        """
        results = {
            'k_values': list(k_range),
            'MAE': [],
            'silhouette': [],
            'dbi': [],
            'irrelevant_clusters': [],
            'wcss': []
        }
        
        min_cluster_size = len(self.X) * threshold_percent / 100
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42).fit(self.X)
            labels = kmeans.labels_
            
            # Calcular métricas
            results['MAE'].append(mean_absolute_error(self.X, kmeans.cluster_centers_[labels]))
            results['silhouette'].append(silhouette_score(self.X, labels))
            results['dbi'].append(davies_bouldin_score(self.X, labels))
            results['wcss'].append(kmeans.inertia_)
            
            # Contar clusters irrelevantes
            counts = np.unique(labels, return_counts=True)[1]
            results['irrelevant_clusters'].append(sum(counts < min_cluster_size))
        
        return results
    
    def plot_metrics(self, results):
        """Grafica las métricas para selección de k"""
        metrics = {
        'MAE': {'color': 'bo-', 'title': 'MAE vs k'},
        'silhouette': {'color': 'go-', 'title': 'Silhouette Score vs k'},
        'dbi': {'color': 'ro-', 'title': 'Davies-Bouldin Index vs k'},
        'irrelevant_clusters': {'color': 'mo-', 'title': 'Clusters Irrelevantes vs k'},
        'wcss': {'color': 'co-', 'title': 'WCSS vs k'}
        }
        save_paths = {
            'MAE': os.path.join(self.current_script_dir, '..', 'img_results', 'MAE_kmeans.png'),
            'silhouette': os.path.join(self.current_script_dir, '..', 'img_results', 'silhouette_kmeans.png'),
            'dbi': os.path.join(self.current_script_dir, '..', 'img_results', 'dbi_kmeans.png'),
            'irrelevant_clusters': os.path.join(self.current_script_dir, '..', 'img_results', 'irrelevant_clusters_kmeans.png'),
            'wcss': os.path.join(self.current_script_dir, '..', 'img_results', 'wcss_kmeans.png')
        }
        for metric, config in metrics.items():
            plt.figure(figsize=(10, 5))
            plt.plot(results['k_values'], results[metric], config['color'], linewidth=2, markersize=8, markerfacecolor='white')
            plt.title(config['title'], pad=15, fontsize=14)
            plt.xlabel('Number of clusters (k)', labelpad=10, fontsize=12)
            plt.ylabel(metric, labelpad=10, fontsize=12)
            plt.xticks(results['k_values'][::2])
            plt.grid(True, linestyle='--', alpha=0.7)
            if save_paths and metric in save_paths:
                plt.savefig(save_paths[metric])
            plt.close()
    
    def train_final_model(self, k, save_path=None):
        """
        Entrena el modelo final con el k seleccionado
        
        Args:
            k: Número de clusters seleccionado
            save_path: Ruta para guardar el modelo
            
        Returns:
            Modelo KMeans entrenado
        """
        self.kmeans_model = KMeans(n_clusters=k, random_state=42).fit(self.X)
        
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(self.kmeans_model, f)
        


    def pipeline(self, k_range=range(2, 30)):
        """
        Pipeline completo simplificado
        
        Args:
            k_range: Rango de valores k a evaluar
        """
        print("Evaluando métricas para diferentes valores de k...")
        results = self.find_optimal_k(k_range)

        
        self.plot_metrics(results)
        
        k_isac_tlp = K_ISAC_TLP()
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', 'k-isac_tlp_kmeans.png')
        final_k = k_isac_tlp.k_ISAC_TLP(
            results['k_values'], 
            results['MAE'], 
            results['irrelevant_clusters'],
            save_path=save_path
        )
        print(f"\nK óptimo sugerido: {final_k} (basado en K-ISAC TLP)")
        
        print("\nEntrenando modelo final...")
        save_path = os.path.join(self.current_script_dir, '..', 'pkls', 'kmeans_model.pkl')
        self.train_final_model(final_k, save_path=save_path)
    

class MySpectralClustering:
    def __init__(self, data_path="dataset/features_StandardScaler.csv"):
        """Inicializa con los datos escalados"""

        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.df = pd.read_csv(data_path, index_col='cups')
        self.X = self.df.values
        self.model = None
    
    def evaluate_parameters(self, k_range=range(2, 15), 
                          gamma_values=[0.1, 0.5, 1.0, 2.0, 5.0],
                          n_neighbors_values=[5, 10, 15, 20, 25]):
        """
        Evalúa diferentes configuraciones de parámetros para Spectral Clustering
        
        Args:
            k_range: Rango de valores k a evaluar
            gamma_values: Valores de gamma para kernel RBF
            n_neighbors_values: Valores de vecinos para affinity nearest_neighbors
            
        Returns:
            DataFrame con resultados de las métricas
        """
        results = []
        
        # Evaluar kernel RBF
        for gamma in gamma_values:
            for k in k_range:
                try:
                    sc = SpectralClustering(n_clusters=k, affinity='rbf', 
                                          gamma=gamma, random_state=42)
                    labels = sc.fit_predict(self.X)
                    
                    if len(np.unique(labels)) == k:
                        results.append({
                            'method': 'RBF',
                            'n_clusters': k,
                            'param': gamma,
                            'silhouette': silhouette_score(self.X, labels),
                            'davies_bouldin': davies_bouldin_score(self.X, labels)
                        })
                except:
                    continue
        
        # Evaluar nearest neighbors
        for n_neighbors in n_neighbors_values:
            for k in k_range:
                try:
                    sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
                                          n_neighbors=n_neighbors, random_state=42)
                    labels = sc.fit_predict(self.X)
                    if len(np.unique(labels)) == k:
                        results.append({
                            'method': 'NearestNeighbors',
                            'n_clusters': k,
                            'param': n_neighbors,
                            'silhouette': silhouette_score(self.X, labels),
                            'davies_bouldin': davies_bouldin_score(self.X, labels)
                        })
                except:
                    continue
        
        return pd.DataFrame(results)
    def plot_results(self, results_df, save_path=None):
        """Visualiza los resultados de las métricas"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Silhouette Score
        for method in results_df['method'].unique():
            subset = results_df[results_df['method'] == method]
            ax1.plot(subset['n_clusters'], subset['silhouette'], 'o-', 
                    label=f"{method} (param={subset['param'].iloc[0]})")
        ax1.set_title('Silhouette Score')
        ax1.legend()
        
        # Davies-Bouldin Index
        for method in results_df['method'].unique():
            subset = results_df[results_df['method'] == method]
            ax2.plot(subset['n_clusters'], subset['davies_bouldin'], 'o-',
                    label=f"{method} (param={subset['param'].iloc[0]})")
        ax2.set_title('Davies-Bouldin Index (lower is better)')
        ax2.legend()
        
        plt.tight_layout()
        #plt.show()
        if save_path:
            fig.savefig(save_path)
        
    
    def train_final_model(self, k, affinity='rbf', gamma=1.0, n_neighbors=10,
                        save_path=None):
        """
        Entrena el modelo final con los parámetros seleccionados
        
        Args:
            k: Número de clusters
            affinity: Tipo de afinidad ('rbf' o 'nearest_neighbors')
            gamma: Parámetro gamma para kernel RBF
            n_neighbors: Número de vecinos para affinity nearest_neighbors
            save_path: Ruta para guardar el modelo

        """
        if affinity == 'rbf':

            self.model = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                gamma=gamma,
                random_state=42
            )
        elif affinity == 'nearest_neighbors':
            self.model = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                n_neighbors=n_neighbors,
                random_state=42
            )
        else:
            self.model = SpectralClustering(
                n_clusters=k,
                affinity=affinity,
                gamma=gamma,
                random_state=42
            )
        self.model.fit(self.X)
        
        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(self.model, f)


    
    def pipeline(self, k_range=range(2, 15), gamma_values=[0.1, 0.5, 1.0, 2.0, 5.0],
                 n_neighbors_values=[5, 10, 15, 20, 25]):
        """
        Pipeline completo simplificado

        """
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', 'spectral_clustering_metrics.png')
        results = self.evaluate_parameters(k_range=k_range, 
                          gamma_values=gamma_values,
                          n_neighbors_values=n_neighbors_values)
        
        self.plot_results(results, save_path)
        print("\nEntrenando modelo final...")

        affinity = 'rbf'
        gamma = 1.0
        k = 3
        print(f"Usando k={k}, affinity={affinity}, gamma={gamma}")
        save_path = os.path.join(self.current_script_dir, '..', 'pkls', 'sc_model.pkl')
        self.model = self.train_final_model(k, affinity=affinity, gamma=gamma, save_path=save_path)

