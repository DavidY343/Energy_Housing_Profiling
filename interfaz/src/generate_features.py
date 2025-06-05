import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple
import os

class EnergyFeatureGenerator:
    """
    Clase para preprocesar datos de consumo energético y generar características agregadas.
    
    Atributos:
        df (pd.DataFrame): DataFrame con los datos de consumo
        features_por_cups (dict): Diccionario con features generadas por CUPS
        df_features_final (pd.DataFrame): DataFrame final con todas las features
    """
    
    def __init__(self, data_path: str = "../dataset/vertical_data.csv"):
        """
        Inicializa la clase cargando los datos.
        
        Args:
            data_path (str): Ruta al archivo CSV con los datos
        """
        self.df = pd.read_csv(data_path, sep=";")
        self.features_por_cups = {}
        self.df_features_final = None
        self.current_script_dir = os.path.dirname(os.path.abspath(__file__))
        
        warnings.filterwarnings("ignore")
    
    @staticmethod
    def obtener_estacion(mes: int) -> str:
        """
        Determina la estación del año basada en el mes.
        
        Args:
            mes (int): Mes del año (1-12)
            
        Returns:
            str: Estación del año ('invierno', 'primavera', 'verano', 'otoño')
        """
        if mes in [12, 1, 2]:
            return 'invierno'
        elif mes in [3, 4, 5]:
            return 'primavera'
        elif mes in [6, 7, 8]:
            return 'verano'
        return 'otoño'
    
    @staticmethod
    def calcular_features(consumos: List[float]) -> Dict[str, float]:
        """
        Calcula diversas características estadísticas para una lista de consumos.
        
        Args:
            consumos (list): Lista de valores de consumo
            
        Returns:
            dict: Diccionario con las características calculadas
        """
        consumos = np.array(consumos)
        if len(consumos) < 2:
            return dict.fromkeys([
                'media', 'std', 'skewness', 'kurtosis',
                'energia_fft', 'frecuencia_dominante',
                'mediana', 'p25', 'p75'
            ], np.nan)

        # Estadísticas básicas
        media = np.mean(consumos)
        std = np.std(consumos)
        skewness = skew(consumos)
        kurt = kurtosis(consumos)

        # Análisis de Fourier
        fft_vals = fft(consumos)
        fft_powers = np.abs(fft_vals)**2
        energia = np.sum(fft_powers) / len(fft_powers)
        frec_dom = np.argmax(fft_powers[1:]) + 1 if len(fft_powers) > 1 else np.nan

        # Percentiles
        return {
            'media': media,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurt,
            'energia_fft': energia,
            'frecuencia_dominante': frec_dom,
            'mediana': np.median(consumos),
            'p25': np.percentile(consumos, 25),
            'p75': np.percentile(consumos, 75)
        }
    
    def preprocesar_datos(self) -> None:
        """
        Realiza el preprocesamiento básico de los datos:
        - Conversión de tipos
        - Extracción de características temporales
        - Asignación de estaciones
        """
        self.df['fecha'] = pd.to_datetime(self.df['fecha'])
        self.df['hora'] = self.df['hora'].astype(int)
        self.df['dia_semana'] = self.df['fecha'].dt.dayofweek
        self.df['es_fin_de_semana'] = self.df['dia_semana'] >= 5
        self.df['mes'] = self.df['fecha'].dt.month
        self.df['estacion'] = self.df['mes'].apply(self.obtener_estacion)
    
    def generar_features_estacionales(self, dividir_por_tipo_dia: bool = False) -> None:
        """
        Genera características agregadas por estación del año.
        
        Args:
            dividir_por_tipo_dia (bool): Si True, divide también entre días laborales y fines de semana
        """
        if dividir_por_tipo_dia:
            df_grouped = self.df.groupby(
                ['cups', 'fecha', 'es_fin_de_semana', 'estacion']
            )['consumo_kWh'].apply(list).reset_index()
            
            # 1. Subanálisis por estación y tipo de día
            for (cups, estacion, es_finde), grupo in df_grouped.groupby(['cups', 'estacion', 'es_fin_de_semana']):
                consumos = sum(grupo['consumo_kWh'].tolist(), [])
                feats = self.calcular_features(consumos)
                tipo_dia = 'finde' if es_finde else 'laboral'
                prefix = f"{estacion}_{tipo_dia}"
                
                if cups not in self.features_por_cups:
                    self.features_por_cups[cups] = {}
                
                for key, val in feats.items():
                    self.features_por_cups[cups][f"{key}_{prefix}"] = val
            
            # 2. Subanálisis ANUAL dividido en laboral/finde
            for (cups, es_finde), grupo in df_grouped.groupby(['cups', 'es_fin_de_semana']):
                consumos = sum(grupo['consumo_kWh'].tolist(), [])
                feats = self.calcular_features(consumos)
                tipo_dia = 'finde' if es_finde else 'laboral'
                prefix = f"anual_{tipo_dia}"
                
                for key, val in feats.items():
                    self.features_por_cups[cups][f"{key}_{prefix}"] = val
        else:
            df_grouped = self.df.groupby(['cups', 'fecha', 'estacion'])['consumo_kWh'].apply(list).reset_index()
            
            # 1. Subanálisis por estación
            for (cups, estacion), grupo in df_grouped.groupby(['cups', 'estacion']):
                consumos = sum(grupo['consumo_kWh'].tolist(), [])
                feats = self.calcular_features(consumos)
                prefix = f"{estacion}"

                if cups not in self.features_por_cups:
                    self.features_por_cups[cups] = {}

                for key, val in feats.items():
                    self.features_por_cups[cups][f"{key}_{prefix}"] = val
        
        # 3. Subanálisis ANUAL completo (sin dividir)
        for cups, grupo in self.df.groupby('cups'):
            consumos = grupo['consumo_kWh'].tolist()
            feats = self.calcular_features(consumos)
            for key, val in feats.items():
                self.features_por_cups[cups][f"{key}_anual"] = val
    
    def crear_dataframe_final(self) -> None:
        """
        Crea el DataFrame final a partir de las features generadas.
        """
        self.df_features_final = pd.DataFrame.from_dict(self.features_por_cups, orient='index')
        self.df_features_final.reset_index(inplace=True)
        self.df_features_final.rename(columns={'index': 'cups'}, inplace=True)
        self.df_features_final.set_index('cups', inplace=True)
    
    def analizar_correlaciones(self):
        """
        Analiza las correlaciones entre features y genera visualizaciones.
        
        """
        correlation_matrix = self.df_features_final.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matriz de Correlación de las Features')
        
        plot_path = os.path.join(self.current_script_dir, '..', 'img_results', 'correlation_matrix.png')
        
        plt.savefig(plot_path)
        #plt.show()
        
        # Triángulo superior para evitar duplicados
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # Identificar pares con baja correlación
        low_correlation_threshold = 0.05
        low_correlated_pairs = [
            (column, row, upper_triangle.loc[row, column]) 
            for column in upper_triangle.columns 
            for row in upper_triangle.index 
            if abs(upper_triangle.loc[row, column]) < low_correlation_threshold
        ]

        print(f"\nPares de variables con baja correlación (abs(corr) < {low_correlation_threshold}):\n")
        for pair in low_correlated_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

        # Identificar pares con alta correlación
        high_correlation_threshold = 0.95
        high_correlated_pairs = [
            (column, row, upper_triangle.loc[row, column]) 
            for column in upper_triangle.columns 
            for row in upper_triangle.index 
            if abs(upper_triangle.loc[row, column]) > high_correlation_threshold
        ]

        print(f"\nPares de variables con ALTA correlación (abs(corr) > {high_correlation_threshold}):\n")
        for pair in high_correlated_pairs:
            print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")
        
    
    def guardar_features(self) -> None:
        """
        Guarda las features generadas en un archivo CSV.
        
        """
        output_dir = os.path.join(self.current_script_dir, '..', 'dataset')
        dataset_path = f"{output_dir}/generated_features.csv"
        self.df_features_final.to_csv(dataset_path, sep=";", index=True)
        print(f"Features guardadas en {dataset_path}")
    
    def resumen_features(self) -> None:
        """
        Muestra un resumen de las features generadas.
        """
        if self.df_features_final is None:
            print("No hay features generadas. Ejecutar generar_features_estacionales() primero.")
            return
        
        num_columnas = self.df_features_final.shape[1]
        print(f"\nEl DataFrame tiene {num_columnas} columnas.")
        
        nombres_columnas = self.df_features_final.columns.tolist()
        
        # Agrupar features por tipo
        columnas_invierno = [col for col in nombres_columnas if re.search(r'_invierno', col)]
        columnas_primavera = [col for col in nombres_columnas if re.search(r'_primavera', col)]
        columnas_verano = [col for col in nombres_columnas if re.search(r'_verano', col)]
        columnas_otoño = [col for col in nombres_columnas if re.search(r'_otoño', col)]
        columnas_anuales = [col for col in nombres_columnas if re.search(r'_anual', col)]
        
        print("\nResumen por categoría:")
        print(f"- Invierno: {len(columnas_invierno)} features")
        print(f"- Primavera: {len(columnas_primavera)} features")
        print(f"- Verano: {len(columnas_verano)} features")
        print(f"- Otoño: {len(columnas_otoño)} features")
        print(f"- Anuales: {len(columnas_anuales)} features")
    
    def pipeline_completo(self, dividir_por_tipo_dia: bool = False) -> None:
        """
        Ejecuta el pipeline completo de preprocesamiento y generación de features.
        
        Args:
            dividir_por_tipo_dia (bool): Si True, divide features entre días laborales y fines de semana
        """
        print("Iniciando pipeline de generación de features...")
        self.preprocesar_datos()
        print("Datos preprocesados")
        
        self.generar_features_estacionales(dividir_por_tipo_dia)
        print("Features estacionales generadas")
        
        self.crear_dataframe_final()
        print("DataFrame final creado")
        
        self.guardar_features()
        print("Features guardadas en archivo CSV")

        self.analizar_correlaciones()
        print("Análisis de correlaciones realizado")

        self.resumen_features()
        print("\nPipeline completado exitosamente!")