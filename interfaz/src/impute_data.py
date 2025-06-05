import pandas as pd
from sklearn.impute import KNNImputer
from fancyimpute import SoftImpute
import warnings
from typing import Tuple, List
import os


class MissingValueHandler:
    """
    Clase para manejar valores faltantes en datos de consumo energético.
    
    Atributos:
        df (pd.DataFrame): DataFrame con los datos de features
        imputed_dfs (dict): Diccionario con DataFrames imputados
        cups_a_eliminar (dict): Diccionario con CUPS a eliminar por estación
    """
    
    def __init__(self, data_path: str = "dataset/generated_features.csv"):
        """
        Inicializa la clase cargando los datos.
        
        Args:
            data_path (str): Ruta al archivo CSV con las features
        """
        self.df = pd.read_csv(data_path, sep=";", index_col='cups')
        self.imputed_dfs = {}
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        warnings.filterwarnings("ignore")
    
    def analizar_nans(self) -> Tuple[bool, List[str], pd.Series]:
        """
        Analiza los valores NaN en el DataFrame.
        
        Returns:
            tuple: (tiene_nans, columnas_con_nans, nans_por_columna)
        """
        tiene_nans = self.df.isnull().any().any()
        columnas_con_nans = []
        nans_por_columna = pd.Series()
        
        if tiene_nans:
            columnas_con_nans = self.df.columns[self.df.isnull().any()].tolist()
            nans_por_columna = self.df.isnull().sum()
            nans_por_columna = nans_por_columna[nans_por_columna > 0]
        
        return tiene_nans, columnas_con_nans, nans_por_columna
    
    def imputar_knn(self, save_path: str = None) -> pd.DataFrame:
        """
        Imputa valores faltantes usando KNNImputer.
            
        """
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(
            imputer.fit_transform(self.df),
            columns=self.df.columns,
            index=self.df.index
        )
        
        self.imputed_dfs['knn'] = df_imputed
        
        if save_path:
            df_imputed.to_csv(save_path)
            print(f"Dataset imputado con KNN guardado en {save_path}")
        
    
    def imputar_soft(self, save_path: str = None) -> pd.DataFrame:
        """
        Imputa valores faltantes usando SoftImpute.

        """
        imputer = SoftImpute()
        df_imputed = pd.DataFrame(
            imputer.fit_transform(self.df),
            columns=self.df.columns,
            index=self.df.index
        )
        
        self.imputed_dfs['soft'] = df_imputed
        
        if save_path:
            df_imputed.to_csv(save_path)
            print(f"Dataset imputado con SoftImpute guardado en {save_path}")
        
    
    def eliminar_filas_con_nans(self, save_path: str = None) -> pd.DataFrame:
        """
        Elimina filas con valores faltantes según criterio predefinido.

        """
        df_drop = self.df.copy()
        
        # Voy a eliminar dos filas (cups) que tienen NaN durante todo el invierno
        cups_con_nan_invierno = ['ceaddbf817fc', 'd0fbcc1108d8']
        df_drop.drop(index=cups_con_nan_invierno, inplace=True, errors='ignore')

        # Voy a eliminar tres filas (cups) que tienen NaN durante todo el otoño
        cups_con_nan_otoño = ['83c7fbada9b4', 'b476034a2e3d', 'ba38270a360e']
        df_drop.drop(index=cups_con_nan_otoño, inplace=True, errors='ignore')
        
        if save_path:
            df_drop.to_csv(save_path)
            print(f"Dataset con filas eliminadas guardado en {save_path}")

    
    def pipeline_completo(self) -> None:
        """
        Ejecuta el pipeline completo de manejo de valores faltantes.
        """
        print("Iniciando análisis de valores faltantes...")
        
        tiene_nans, columnas_con_nans, nans_por_columna = self.analizar_nans()
        
        if not tiene_nans:
            print("✅ El dataset no contiene valores faltantes.")
            return
        
        print("\nResumen de valores faltantes:")
        print(f"- Columnas con NaN: {columnas_con_nans}")
        print("\nNúmero de NaNs por columna:")
        print(nans_por_columna)
        
        print("\nAplicando métodos de imputación...")

        save_path = os.path.join(self.current_script_dir, '..', 'dataset', 'features_KNNImputer.csv')
        self.imputar_knn(save_path)

        save_path = os.path.join(self.current_script_dir, '..', 'dataset', 'features_SoftImpute.csv')
        self.imputar_soft(save_path)

        save_path = os.path.join(self.current_script_dir, '..', 'dataset', 'features_drop.csv')
        self.eliminar_filas_con_nans(save_path)
        
        print("\n✅ Pipeline completado exitosamente!")
        print("DataFrames disponibles en el atributo imputed_dfs:")
        print("- knn: Imputación con KNN")
        print("- soft: Imputación con SoftImpute")
        print("- drop: Eliminación de filas problemáticas")