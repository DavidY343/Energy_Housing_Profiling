import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import os


class FeatureScaler:
    """
    Clase para escalado de características de consumo energético.
    
    Atributos:
        df (pd.DataFrame): DataFrame con los datos originales
        scaler (StandardScaler): Instancia del escalador
        scaler (MinMaxScaler): Instancia del escalador
        scaled_df (pd.DataFrame): DataFrame con las características escaladas
    """
    
    def __init__(self, data_path: str = "dataset/features_KNNImputer.csv"):
        """
        Inicializa la clase cargando los datos.
        
        Args:
            data_path (str): Ruta al archivo CSV con las features
        """
        self.df = pd.read_csv(data_path, index_col='cups')
        self.standar_scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.scaled_df = None
        warnings.filterwarnings("ignore")
    
    def aplicar_min_max_scaler(self, save_path: str = None):
        """
        Aplica MinMaxScaler a los datos.
        
        Args:
            save_path (str): Ruta para guardar los datos escalados (opcional)
        """
        features = self.df.values
        scaled_features = self.min_max_scaler.fit_transform(features)
        
        self.scaled_df = pd.DataFrame(
            scaled_features,
            index=self.df.index,
            columns=self.df.columns
        )
        
        if save_path:
            self.scaled_df.to_csv(save_path)
            print(f"\nDatos escalados guardados en {save_path}")


    def aplicar_standard_scaler(self, save_path: str = None):
        """
        Aplica StandardScaler a los datos.
        
        Args:
            save_path (str): Ruta para guardar los datos escalados (opcional)
        """
        features = self.df.values
        scaled_features = self.standar_scaler.fit_transform(features)
        
        self.scaled_df = pd.DataFrame(
            scaled_features,
            index=self.df.index,
            columns=self.df.columns
        )
        
        if save_path:
            self.scaled_df.to_csv(save_path)
            print(f"\nDatos escalados guardados en {save_path}")
    
    def pipeline_completo(self) -> None:
        """
        Ejecuta el pipeline completo de escalado.
        
        Args:
            save_path (str): Ruta para guardar los datos escalados
        """
        print("Iniciando pipeline de escalado de características...")

        save_path = os.path.join(self.current_script_dir, '..', 'dataset', 'features_MinMaxScaler.csv')
        self.aplicar_min_max_scaler(save_path)

        save_path = os.path.join(self.current_script_dir, '..', 'dataset', 'features_StandardScaler.csv')
        self.aplicar_standard_scaler(save_path)

        print("\nPipeline completado exitosamente!")