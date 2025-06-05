import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from matplotlib import cm
import os
class ClusterVisualizer:
    def __init__(self, features_path="dataset/features_StandardScaler.csv", 
                 original_data_path="../data/vertical_preprocessed_data_2.csv",
                 model_type="kmeans"):
        """
        Inicializa el visualizador con los datos y modelo
        
        Args:
            features_path: Ruta al archivo de features escaladas
            original_data_path: Ruta a los datos originales de consumo
            model_type: Tipo de modelo ('kmeans' o 'bkmeans' o 'sc')
        """
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.df_features = pd.read_csv(features_path, index_col='cups')
        self.X = self.df_features.values
        self.model_type = model_type
        self.load_model()
        self.prepare_original_data(original_data_path)
        
    def load_model(self):
        """Carga el modelo de clustering entrenado"""
        model_path = os.path.join(self.current_script_dir, '..', 'pkls', f'{self.model_type}_model.pkl')
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        self.labels = self.model.labels_
        self.best_k = self.model.n_clusters
        self.cluster_colors = cm.get_cmap('tab10', self.best_k)
        
    def prepare_original_data(self, original_data_path):
        """Prepara los datos originales de consumo con los clusters asignados"""
        # Filtrar CUPS problemáticos
        df_original = pd.read_csv(original_data_path, sep=";")
        
        # Asignar clusters
        df_clusters = self.df_features.reset_index()[['cups']].copy()
        df_clusters['cluster'] = self.labels
        self.df_final = pd.merge(df_original, df_clusters, on='cups', how='left')
        
    def plot_cluster_centers(self, save_path=None):
        """Visualiza los centros de los clusters"""
        plt.figure(figsize=(25, 10))
        
        for i in range(self.best_k):
            plt.plot(self.model.cluster_centers_[i], 
                     label=f'Cluster {i}',
                     marker='o',
                     color=self.cluster_colors(i))
        
        plt.xticks(np.arange(len(self.df_features.columns)), 
                   self.df_features.columns, 
                   rotation=45, ha='right')
        plt.title(f'Centros de Cluster ({self.best_k} clusters)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()
    
    def plot_pca_2d(self, save_path=None):
        """Visualización 2D de los clusters usando PCA"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        plt.figure(figsize=(10, 6))
        for cluster in range(self.best_k):
            plt.scatter(X_pca[self.labels == cluster, 0],
                        X_pca[self.labels == cluster, 1],
                        label=f'Cluster {cluster}',
                        color=self.cluster_colors(cluster),
                        alpha=0.7)
        
        plt.title('Visualización 2D de Clusters (PCA)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()


    def plot_pca_3d(self, save_path=None):
        """Visualización 3D de los clusters usando PCA"""
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.X)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster in range(self.best_k):
            ax.scatter(X_pca[self.labels == cluster, 0],
                       X_pca[self.labels == cluster, 1],
                       X_pca[self.labels == cluster, 2],
                       label=f'Cluster {cluster}',
                       color=self.cluster_colors(cluster),
                       alpha=0.7)
        
        ax.set_title('Visualización 3D de Clusters (PCA)')
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        ax.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()

    
    def plot_cluster_distribution(self, save_path=None):
        """Muestra la distribución de CUPS por cluster"""
        cluster_counts = pd.Series(self.labels).value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(cluster_counts.index, cluster_counts.values,
                      color=[self.cluster_colors(i) for i in cluster_counts.index])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 2, str(height), ha='center')
        
        plt.title('Distribución de CUPS por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Número de CUPS')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()


    def plot_cluster_series(self, cluster_num, y_lim=(0, 8), save_path=None):
        """Grafica series temporales de un cluster específico"""
        cluster_data = self.df_final[self.df_final['cluster'] == cluster_num]
        cluster_data['datetime'] = pd.to_datetime(cluster_data['fecha']) + pd.to_timedelta(cluster_data['hora'], unit='h')
        
        plt.figure(figsize=(12, 6))
        
        # Series individuales
        for cups_id in cluster_data['cups'].unique():
            cups_series = cluster_data[cluster_data['cups'] == cups_id]
            plt.plot(cups_series['datetime'], cups_series['consumo_kWh'], 
                     color='lightgray', alpha=0.15, linewidth=0.8)
        
        # Media del cluster
        cluster_mean = cluster_data.groupby('datetime')['consumo_kWh'].mean()
        plt.plot(cluster_mean.index, cluster_mean.values, 
                 label=f'Media Cluster {cluster_num}', linewidth=3, color='red')
        
        plt.title(f'Patrones de Consumo - Cluster {cluster_num}')
        plt.xlabel('Fecha y Hora')
        plt.ylabel('Consumo (kWh)')
        plt.ylim(y_lim)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()

    def save_cluster_assignments(self, save_path):
        """Guarda las asignaciones de clusters a archivo"""
        df_clusters = pd.DataFrame({
            'CUPS': self.df_features.index,
            'CLUSTER': self.labels
        })
        df_clusters.to_csv(save_path, index=False)
    
    def analyze_clusters(self):
        """Proporciona un análisis interpretativo de los clusters"""
        centroids = pd.DataFrame(self.model.cluster_centers_, 
                                columns=self.df_features.columns)
        
        cluster_analysis = []

        for i in range(self.best_k):
            top_features = centroids.iloc[i].sort_values(ascending=False).head(5)
            
            bottom_features = centroids.iloc[i].sort_values().head(5)
            
            analysis = {
                'Cluster': f'Cluster {i}',
                'Tamaño': sum(self.model.labels_ == i),
                'Características destacadas': ', '.join([f"{feat} ({val:.2f})" 
                                            for feat, val in top_features.items()]),
                'Características bajas': ', '.join([f"{feat} ({val:.2f})" 
                                        for feat, val in bottom_features.items()])
            }
            cluster_analysis.append(analysis)

        analysis_df = pd.DataFrame(cluster_analysis)
        print(analysis_df.to_string(index=False))
        for i in range(self.best_k):
            cluster = centroids.iloc[i]
            size = sum(self.model.labels_ == i)
            percentage = size/len(self.model.labels_)*100
            
            print(f"\nCluster {i} ({size} viviendas, {percentage:.1f}% del total):")
            
            # 1. Análisis por estaciones
            estaciones = ['invierno', 'otoño', 'primavera', 'verano']
            
            # Encontrar la estación con mayor consumo medio
            medias_estacionales = {est: cluster[f'media_{est}'] for est in estaciones}
            estacion_max = max(medias_estacionales, key=medias_estacionales.get)
            print(f"- Mayor consumo en {estacion_max} (media={medias_estacionales[estacion_max]:.2f})")
            
            # 2. Variabilidad (std)
            stds = {est: cluster[f'std_{est}'] for est in estaciones}
            estacion_mas_variable = max(stds, key=stds.get)
            print(f"- Mayor variabilidad en {estacion_mas_variable} (std={stds[estacion_mas_variable]:.2f})")
            
            # 3. Patrones de forma (skewness)
            skews = {est: cluster[f'skewness_{est}'] for est in estaciones}
            estacion_skew_positivo = max(skews, key=skews.get)
            estacion_skew_negativo = min(skews, key=skews.get)
            print(f"- Distribución más asimétrica positiva en {estacion_skew_positivo}")
            print(f"- Distribución más asimétrica negativa en {estacion_skew_negativo}")
            
            # 4. Energía en frecuencia dominante (FFT)
            energias = {est: cluster[f'energia_fft_{est}'] for est in estaciones}
            estacion_energia_max = max(energias, key=energias.get)
            print(f"- Patrón estacional más marcado en {estacion_energia_max}")
            
            # 5. Comparación anual vs estacional
            if cluster['media_anual'] > np.mean([cluster[f'media_{est}'] for est in estaciones]):
                print("- Consumo anual superior al promedio estacional")
            else:
                print("- Consumo anual inferior al promedio estacional")

    def pipeline(self):
        """Ejecuta todo el pipeline de visualización"""
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_centers_{self.model_type}.png')
        self.plot_cluster_centers(save_path)
        
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_pca2d_{self.model_type}.png')
        self.plot_pca_2d(save_path)

        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_pca3d_{self.model_type}.png')
        self.plot_pca_3d(save_path)
        
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_distribution_{self.model_type}.png')
        self.plot_cluster_distribution(save_path)
        
        for i in range(self.best_k):
            save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_series_{i}_{self.model_type}.png')
            self.plot_cluster_series(i, save_path=save_path)
        
        save_path = os.path.join(self.current_script_dir, '..', 'dataset', f'cluster_cups_{self.model_type}.csv')
        self.save_cluster_assignments(save_path)
        self.analyze_clusters()



class SpectralClusterVisualizer:
    def __init__(self, features_path="dataset/features_StandardScaler.csv", 
                 original_data_path="../data/vertical_preprocessed_data_2.csv"):
        """
        Inicializa el visualizador para Spectral Clustering
        
        Args:
            features_path: Ruta a las features escaladas
            original_data_path: Ruta a los datos originales de consumo
        """
        # Cargar datos y modelo
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.df_features = pd.read_csv(features_path, index_col='cups')
        self.X = self.df_features.values
        
        # Cargar modelo entrenado
        model_path = os.path.join(self.current_script_dir, '..', 'pkls', 'sc_model.pkl')
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        self.labels = self.model.labels_
        self.best_k = len(np.unique(self.labels))
        
        # Configurar colores
        self.colors = {
            0: 'blue',
            1: 'yellow', 
            2: 'red'
        }
        self.cvec = [self.colors[label] for label in self.labels]
        
        # Preparar datos originales
        self.prepare_original_data(original_data_path)
    
    def prepare_original_data(self, original_data_path):
        """Prepara los datos originales con asignaciones de cluster"""
        df_original = pd.read_csv(original_data_path, sep=";")
        
        # Asignar clusters
        df_clusters = self.df_features.reset_index()[['cups']].copy()
        df_clusters['cluster'] = self.labels
        self.df_final = pd.merge(df_original, df_clusters, on='cups', how='left')
    
    def plot_pca_2d(self, save_path=None):
        """Visualización 2D con PCA"""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        
        plt.figure(figsize=(10, 8))

        scatters = [
            plt.scatter([], [], color='blue', label='Cluster 0', alpha=0.7),
            plt.scatter([], [], color='yellow', label='Cluster 1', alpha=0.7),
            plt.scatter([], [], color='red', label='Cluster 2', alpha=0.7)
        ]
        
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.cvec, alpha=0.7, edgecolors='w', s=50)
        
        plt.title('Spectral Clustering - Visualización 2D')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend(handles=scatters)
        plt.grid(True, linestyle='--', alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()
    
    def plot_pca_3d(self, save_path=None):
        """Visualización 3D con PCA"""
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.X)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                  c=self.cvec, alpha=0.7, edgecolors='w', s=50)
        
        # Configurar leyenda
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Cluster 0',
                      markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1',
                      markerfacecolor='yellow', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Cluster 2',
                      markerfacecolor='red', markersize=10)
        ]
        
        ax.set_title('Spectral Clustering - Visualización 3D')
        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2') 
        ax.set_zlabel('Componente Principal 3')
        ax.legend(handles=legend_elements)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()
    
    def plot_cluster_distribution(self, save_path=None):
        """Muestra distribución de puntos por cluster"""
        cluster_counts = pd.Series(self.labels).value_counts().sort_index()
        
        plt.figure(figsize=(12, 7))
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
        
        bars = plt.bar(
            cluster_counts.index.astype(str),
            cluster_counts.values,
            color=colors,
            edgecolor='black',
            alpha=0.85
        )
        
        plt.title('Distribución de CUPS por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Número de CUPS')
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        
        # Añadir conteos en las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 2, 
                    f"{int(height)}", ha='center', fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()
    
    def plot_cluster_series(self, cluster_num, y_lim=(0, 8), save_path=None):
        """Grafica series temporales de un cluster específico"""
        cluster_data = self.df_final[self.df_final['cluster'] == cluster_num]
        cluster_data['datetime'] = pd.to_datetime(cluster_data['fecha']) + pd.to_timedelta(cluster_data['hora'], unit='h')
        
        plt.figure(figsize=(12, 6))
        
        # Series individuales (transparentes)
        for cups_id in cluster_data['cups'].unique():
            cups_series = cluster_data[cluster_data['cups'] == cups_id]
            plt.plot(cups_series['datetime'], cups_series['consumo_kWh'], 
                    color='lightgray', alpha=0.15, linewidth=0.8)
        
        # Media del cluster
        cluster_mean = cluster_data.groupby('datetime')['consumo_kWh'].mean()
        plt.plot(cluster_mean.index, cluster_mean.values, 
                label=f'Cluster {cluster_num}', linewidth=3, color='red')
        
        plt.title(f'Patrones de Consumo - Cluster {cluster_num}')
        plt.xlabel('Fecha y Hora')
        plt.ylabel('Consumo (kWh)')
        plt.ylim(y_lim)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        #plt.show()
    
    def save_results(self, save_path):
        """Guarda las asignaciones de clusters"""
        # Guardar asignaciones CUPS-cluster
        df_clusters = pd.DataFrame({
            'CUPS': self.df_features.index,
            'CLUSTER': self.labels
        })
        df_clusters.to_csv(save_path, index=False)

    
    def pipeline(self):
        """Ejecuta todas las visualizaciones y guarda resultados"""
        print(f"🔍 Visualizando resultados para Spectral Clustering (k={self.best_k})")
        
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_pca2d_sc.png')
        self.plot_pca_2d(save_path)
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_pca3d_sc.png')
        self.plot_pca_3d(save_path)
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_distribution_sc.png')
        self.plot_cluster_distribution(save_path)
        
        for cluster in range(self.best_k):
            save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'cluster_series_{cluster}_sc.png')
            self.plot_cluster_series(cluster, save_path=save_path)
        
        save_path = os.path.join(self.current_script_dir, '..', 'dataset', f'cluster_cups_sc.csv')
        self.save_results(save_path)
        print("✅ Resultados guardados en archivos CSV")