import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import pickle
import os
import numpy as np

class MyBinaryTree:
    def __init__(self, features_path="dataset/features_KNNImputer.csv", 
                 scaled_features_path="dataset/features_StandardScaler.csv",
                 model_type="kmeans"):
        """
        Inicializa el intérprete de clusters
        
        Args:
            features_path: Ruta a features originales (sin escalar)
            scaled_features_path: Ruta a features escaladas
            model_type: Tipo de modelo ('kmeans', 'bkmeans' o 'sc')
        """
        # Cargar datos
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.df_features = pd.read_csv(features_path, index_col='cups')
        self.df_scaled = pd.read_csv(scaled_features_path, index_col='cups')
        self.model_type = model_type
        # Cargar modelo de clustering
        self.load_model(model_type)
        
    def load_model(self, model_type):
        """Carga el modelo de clustering especificado"""

        model_paths = {
            'kmeans': os.path.join(self.current_script_dir, '..', 'pkls', 'kmeans_model.pkl'),
            'bkmeans':  os.path.join(self.current_script_dir, '..', 'pkls', 'bkmeans_model.pkl'),
            'sc':  os.path.join(self.current_script_dir, '..', 'pkls', 'sc_model.pkl'),
        }
        
        if model_type not in model_paths:
            raise ValueError("Modelo no válido. Opciones: 'kmeans', 'bkmeans', 'sc'")
            
        with open(model_paths[model_type], "rb") as f:
            self.model = pickle.load(f)
            
        self.labels = self.model.labels_
        self.best_k = len(np.unique(self.labels))
        
    def train_decision_tree(self, max_depth=5):
        """
        Entrena un árbol de decisión para interpretar los clusters
        
        Args:
            max_depth: Profundidad máxima del árbol
            
        Returns:
            Árbol de decisión entrenado
        """
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42
        )
        self.tree.fit(self.df_features.values, self.labels)
        return self.tree
        
    def plot_decision_tree(self, figsize=(20, 20), fontsize=8, save_path=None):
        """Visualiza el árbol de decisión"""
        plt.figure(figsize=figsize)
        plot_tree(
            self.tree,
            feature_names=self.df_features.columns,
            class_names=[f'Cluster {i}' for i in range(self.best_k)],
            filled=True,
            rounded=True,
            proportion=False,
            fontsize=fontsize
        )
        plt.title('Árbol de Decisión para Interpretación de Clusters', fontsize=14)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self, save_path=None):
        """Muestra la importancia de las características"""
        importance = pd.DataFrame({
            'Feature': self.df_features.columns,
            'Importance': self.tree.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        importance = importance[importance['Importance'] > 0]
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['Feature'], importance['Importance'])
        plt.xlabel('Importancia')
        plt.title('Importancia de Características para Asignación de Clusters')
        plt.gca().invert_yaxis()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        
    def get_decision_rules(self):
        """Devuelve las reglas de decisión en formato texto"""
        return export_text(
            self.tree,
            feature_names=list(self.df_features.columns),
            spacing=3,
            decimals=2
        )
        
    def pipeline(self):
        """
        Pipeline completo para interpretar clusters:
        1. Entrena árbol de decisión
        2. Muestra visualización
        3. Muestra importancia de features
        4. Devuelve reglas de decisión
        """

        print(f"🔍 Interpretando {self.best_k} clusters...")
        
        # Entrenar árbol
        self.train_decision_tree(max_depth=5)
        
        # Visualizar
        print("\n🌳 Visualizando árbol de decisión...")

        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'decision_tree_{self.model_type}.png')
        if self.model_type == 'sc':
            self.plot_decision_tree(figsize=(20, 20), fontsize=10, save_path=save_path)
        elif self.model_type == 'bkmeans':
            self.plot_decision_tree(figsize=(40, 20), fontsize=8,  save_path=save_path)
        else:
            self.plot_decision_tree(figsize=(40, 20), fontsize=6, save_path=save_path)
        
        
        # Importancia de features
        print("\n📊 Mostrando importancia de características...")
        save_path = os.path.join(self.current_script_dir, '..', 'img_results', f'feature_importance_{self.model_type}.png')           
        self.plot_feature_importance(save_path=save_path)
        # Reglas de decisión
        print("\n📝 Reglas de decisión:")
        print(self.get_decision_rules())
