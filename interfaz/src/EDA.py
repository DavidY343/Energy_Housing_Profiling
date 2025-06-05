
#  Exploratory Data Analysis (EDA)

import pandas as pd
import os
from datetime import datetime, timedelta
from textwrap import dedent
import re
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ##  - Parte 1
# 
# ### 📌 Objetivo
# 
# En esta primera etapa del EDA, se busca consolidar todos los archivos `.csv` disponibles en una única estructura de datos y realizar un análisis de **completitud** para evaluar la calidad temporal de los registros de consumo eléctrico.
# 
# 
# ### 1. Concatenación de Archivos
# 
# Se cargan y concatenan todos los archivos individuales que contienen datos horarios de consumo eléctrico por vivienda. Esta etapa genera un único DataFrame llamado `merged_df`, que será la base para futuros análisis y modelos.
# 
# El resultado se guarda en `../data/vertical_raw_data_2.csv`.
# 
# ### 2. Análisis de Completitud de Datos
# 
# Por cada archivo original se verifica:
# - Si las fechas en el nombre del archivo coinciden con los datos reales.
# - Si existen **fechas faltantes** dentro del rango temporal.
# - Si hay **días incompletos** (menos de 24 horas).
# - Si hay **errores** de lectura o formato.
# 
# Se genera:
# - Un reporte de texto con los resultados para cada archivo.
# - Gráficos visuales por archivo que muestran la completitud diaria y mensual.
# - Un resumen estadístico al finalizar el procesamiento.
# 
# Los resultados se guardan en `../data_analysis_results/`.
# 
# ## 🔧 Limpieza de días incompletos
# 
# Además del análisis, se realizan acciones de limpieza para mejorar la calidad del dataset:
# - Si un día tiene menos de 24 registros, se **rellenan** las horas faltantes con `ffill` o `bfill`.
# - Si un día tiene **más de 24 registros**, se eliminan los duplicados conservando los primeros por hora.
# 

class EDAEnergyConsumption:
    """
    Clase para realizar análisis exploratorio de datos (EDA) de archivos de consumo eléctrico.
    
    Atributos:
        folder_path (str): Ruta a la carpeta con archivos CSV de consumo
        output_dir (str): Directorio para guardar resultados del análisis
        merged_output_path (str): Ruta para guardar el dataset consolidado
        summary_stats (dict): Estadísticas resumen del análisis
        contador_modificaciones (dict): Registro de modificaciones realizadas
        total_modificaciones (int): Total de modificaciones realizadas
        all_dataframes (list): Lista de DataFrames procesados
    """
    
    def __init__(self, folder_path="../data_raw/", output_dir="../data_analysis_results/", 
                 merged_output_path="dataset/vertical_data.csv"):
        """
        Inicializa la clase con las rutas de entrada/salida.
        
        Args:
            folder_path (str): Ruta a los archivos CSV de entrada
            output_dir (str): Directorio para guardar reportes
            merged_output_path (str): Ruta para el dataset consolidado
        """
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.merged_output_path = merged_output_path
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.merged_output_path), exist_ok=True)
        

        self.summary_stats = {
            'total_files': 0,
            'files_with_date_mismatch': 0,
            'files_with_missing_dates': 0,
            'files_with_incomplete_days': 0,
            'files_with_errors': 0,
            'cups_mismatches': defaultdict(int)
        }
        
        self.contador_modificaciones = {}
        self.total_modificaciones = 0
        self.all_dataframes = []
        
        warnings.filterwarnings("ignore")
    
    @staticmethod
    def extract_dates_from_filename(filename):
        """
        Extrae fechas del nombre de archivo.
        
        Args:
            filename (str): Nombre del archivo
            
        Returns:
            tuple: (fecha_inicio, fecha_fin) o (None, None) si no se encuentran
        """
        date_matches = re.findall(r'(\d{2}-\d{2}-\d{4})', filename)
        if len(date_matches) >= 2:
            try:
                return (datetime.strptime(date_matches[0], "%d-%m-%Y").date(),
                        datetime.strptime(date_matches[1], "%d-%m-%Y").date())
            except ValueError:
                pass
        return None, None
    
    @staticmethod
    def extract_cups_from_filename(filename):
        """
        Extrae el código CUPS del nombre de archivo.
        
        Args:
            filename (str): Nombre del archivo
            
        Returns:
            str: Código CUPS o "DESCONOCIDO" si no se encuentra
        """
        cups_match = re.match(r'^([a-f0-9]+)', filename)
        return cups_match.group(1) if cups_match else "DESCONOCIDO"
    
    def fix_incomplete_days(self, df):
        """
        Corrige días incompletos en el DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame con datos de consumo
            
        Returns:
            pd.DataFrame: DataFrame corregido
        """
        fixed_rows = []
        for date, group in df.groupby('fecha'):
            if len(group) < 24:
                full_day = pd.DataFrame({'hora': list(range(24))})
                merged = full_day.merge(group, on='hora', how='left')
                merged['fecha'] = date
                merged.sort_values(by='hora', inplace=True)
                merged.ffill(inplace=True)
                merged.bfill(inplace=True)
                fixed_rows.append(merged)
            elif len(group) > 24:
                trimmed = group.sort_values(by='hora').head(24)
                fixed_rows.append(trimmed)
            else:
                fixed_rows.append(group)
        return pd.concat(fixed_rows, ignore_index=True)
    
    def formatear_hora(self, hora, cups=None):
        """
        Formatea la hora a un formato estándar.
        
        Args:
            hora (str): Hora a formatear
            cups (str): Código CUPS asociado (opcional)
            
        Returns:
            str: Hora formateada
        """
        hora_original = hora
        hora = str(hora).strip()
        modificado = False
        
        # Convertir '1:00' a '01:00', etc.
        if len(hora.split(':')[0]) == 1:
            hora = '0' + hora
            modificado = True
        
        # Eliminar '24:00:00' y '24:00' a '00:00'
        if hora == '24:00:00' or hora == '24:00':
            hora = '00:00'
        
        # Registrar modificaciones
        if modificado and cups is not None:
            self.contador_modificaciones[cups] = self.contador_modificaciones.get(cups, 0) + 1
            self.total_modificaciones += 1
            #print(f"Advertencia: Hora modificada de '{hora_original}' a '{hora}' para el CUPS: {cups}")
        
        return hora
    
    @staticmethod
    def limpiar_consumo(consumo):
        """
        Limpia y convierte el valor de consumo a float.
        
        Args:
            consumo (str): Valor de consumo
            
        Returns:
            float: Valor convertido o NaN si hay error
        """
        try:
            consumo = str(consumo).replace(',', '.')
            return float(consumo)
        except ValueError:
            print("HUBO ERROR")
            return float('nan')
    
    def generate_report(self, file):
        """
        Genera un reporte de análisis para un archivo específico.
        
        Args:
            file (str): Nombre del archivo a analizar
        """
        cups = self.extract_cups_from_filename(file)
        file_path = os.path.join(self.folder_path, file)
        report_name = os.path.splitext(file)[0] + "_report.txt"
        report_path = os.path.join(self.output_dir, report_name)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(dedent(f"""\
            ====================================
            ANÁLISIS DE COMPLETITUD DE DATOS
            Archivo: {file}
            CUPS: {cups}
            ====================================
            """))

            try:
                df = pd.read_csv(file_path, sep=';', dtype={"cups": str})

                # Procesamiento de fechas
                df['fecha'] = pd.to_datetime(df['fecha'], format='mixed', dayfirst=True, errors='coerce').dt.strftime('%d/%m/%Y')
                df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y').dt.date
                df['consumo_kWh'] = df['consumo_kWh'].apply(self.limpiar_consumo)

                # Procesamiento de horas
                df['hora'] = df.apply(
                    lambda row: self.formatear_hora(
                        row['hora'], 
                        cups=row['cups'] if 'cups' in df.columns else None
                    ),
                    axis=1
                )
                
                # Sección de resumen de modificaciones
                f.write(dedent(f"""\
                
                RESUMEN DE FORMATEO DE HORA:
                ====================================
                Modificaciones por CUPS: {self.contador_modificaciones.get(cups, 0)}
                ====================================
                """))
                
                df['hora'] = pd.to_datetime(df['hora'], format='%H:%M', errors='coerce').dt.hour

                # Verificación de rangos de fechas
                expected_start, expected_end = self.extract_dates_from_filename(file)
                if expected_start is None:
                    raise ValueError("No se pudieron extraer fechas del nombre del archivo")

                min_date = df['fecha'].min()
                max_date = df['fecha'].max()
                
                start_mismatch = min_date != expected_start
                end_mismatch = max_date != expected_end
                
                if start_mismatch or end_mismatch:
                    self.summary_stats['files_with_date_mismatch'] += 1
                    self.summary_stats['cups_mismatches'][cups] += 1

                    start_diff = (min_date - expected_start).days if start_mismatch else 0
                    end_diff = (expected_end - max_date).days if end_mismatch else 0
                    
                    f.write(dedent(f"""\
                    ⚠️ DISCREPANCIA DETECTADA:
                    • Esperado (según nombre): {expected_start} a {expected_end}
                    • Encontrado en datos:    {min_date} a {max_date}
                    • Días de desfase:
                      - Inicio: {abs(start_diff)} días ({'antes' if start_diff < 0 else 'después'})
                      - Fin: {abs(end_diff)} días ({'antes' if end_diff > 0 else 'después'})
                    """))
                else:
                    f.write("✅ El rango coincide con lo esperado\n")

                # Verificación de fechas faltantes
                all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
                missing_dates = all_dates[~all_dates.isin(df['fecha'])]
                if len(missing_dates) > 0:
                    self.summary_stats['files_with_missing_dates'] += 1
                    f.write(f"\n Fechas faltantes ({len(missing_dates)}):\n")
                    for date in missing_dates:
                        f.write(f"  - {date.date()}\n")
                else:
                    f.write("\n✅ No hay fechas faltantes en el rango.\n")

                # Verificación de días incompletos
                hours_per_day = df.groupby('fecha')['hora'].count()
                incomplete_days = hours_per_day[(hours_per_day != 24)]
                if len(incomplete_days) > 0:
                    self.summary_stats['files_with_incomplete_days'] += 1
                    f.write(f"\n⚠️ Días con horas incompletas ({len(incomplete_days)}):\n")
                    for date, count in incomplete_days.items():
                        f.write(f"  - {date}: {count}/24 horas\n")
                else:
                    f.write("\n✅ Todos los días tienen 24 registros horarios.\n")

                # Cálculo de completitud
                total_days = len(all_dates)
                complete_days = total_days - len(missing_dates) - len(incomplete_days)
                completeness_percentage = (complete_days / total_days) * 100

                f.write("\n[RESUMEN DE COMPLETITUD]\n")
                f.write(f"• Total de días esperados: {total_days}\n")
                f.write(f"• Días completos (24 horas): {complete_days}\n")
                f.write(f"• Días incompletos: {len(incomplete_days)}\n")
                f.write(f"• Días faltantes: {len(missing_dates)}\n")
                f.write(f"• Completitud: {completeness_percentage:.2f}%\n")

                # Corregir días incompletos y agregar al listado
                df_fixed = self.fix_incomplete_days(df)
                self.all_dataframes.append(df_fixed)

            except Exception as e:
                self.summary_stats['files_with_errors'] += 1
                f.write(f"\n❌ ERROR EN EL PROCESAMIENTO:\n{str(e)}\n")
    
    def generate_summary(self):
        """
        Genera un archivo de resumen con todas las estadísticas del análisis.
        """
        summary_path = os.path.join(self.output_dir, "summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== RESUMEN DE MODIFICACIONES ===\n\n")
            f.write(f"Total de horas modificadas: {self.total_modificaciones}\n")
            for cups, count in self.contador_modificaciones.items():
                f.write(f"CUPS {cups}: {count} veces\n")
            
            f.write("\n=== RESUMEN DE ANÁLISIS ===\n\n")
            f.write(f"Total de archivos procesados: {self.summary_stats['total_files']}\n")
            f.write(f"Archivos con discrepancias de fecha: {self.summary_stats['files_with_date_mismatch']}\n")
            f.write(f"Archivos con fechas faltantes: {self.summary_stats['files_with_missing_dates']}\n")
            f.write(f"Archivos con días incompletos: {self.summary_stats['files_with_incomplete_days']}\n")
            f.write(f"Archivos con errores: {self.summary_stats['files_with_errors']}\n")
            f.write(f"\nAnálisis completado. Resultados guardados en: {self.output_dir}\n")
    
    def process_files(self):
        """
        Procesa todos los archivos CSV en el directorio de entrada.
        """
        for file in os.listdir(self.folder_path):
            if file.endswith(".csv"):
                self.summary_stats['total_files'] += 1
                print(f"Procesando: {file}")
                self.generate_report(file)
        
        if self.all_dataframes:
            merged_df = pd.concat(self.all_dataframes, ignore_index=True)
            merged_df.to_csv(self.merged_output_path, index=False, sep=";")
        
        self.generate_summary()

        print("\n--- Resumen de modificaciones ---")
        print(f"Total de horas modificadas: {self.total_modificaciones}")
        for cups, count in self.contador_modificaciones.items():
            print(f"CUPS {cups}: {count} veces")

        print("\nResumen de análisis:")
        print(f"Total de archivos procesados: {self.summary_stats['total_files']}")
        print(f"Archivos con discrepancias de fecha: {self.summary_stats['files_with_date_mismatch']}")
        print(f"Archivos con fechas faltantes: {self.summary_stats['files_with_missing_dates']}")
        print(f"Archivos con días incompletos: {self.summary_stats['files_with_incomplete_days']}")
        print(f"Archivos con errores: {self.summary_stats['files_with_errors']}")
        print(f"\nAnálisis completado. Resultados guardados en: {self.output_dir}")

    def explore_data_analysis(self, merged_output_path):
        
        df = pd.read_csv(merged_output_path, sep=';')
        print(df.head())
        print(df.info())
        print(df.describe())
        print(f"Las clases unicas de: {df["metodoObtencion"].unique()}")
        print(f"Las clases unicas de: {df["energiaVertida_kWh"].unique()}")

        df.drop(columns=['metodoObtencion'], inplace=True)
        df.drop(columns=['energiaVertida_kWh'], inplace=True)
        print(df.head())

        print(df.dtypes)
        print(df.head())

        nan_check = df.isna().sum()

        print("Número de valores NaN en cada columna:")
        print(nan_check)
        df.to_csv(merged_output_path, sep=';', index=False)
        # A continuación, realizaremos un análisis de la serie temporal para viviendas al azar. Esto nos permitirá observar cómo se distribuye el consumo eléctrico a lo largo del tiempo y detectar posibles patrones, tendencias o picos en su comportamiento. Utilizaremos Matplotlib para visualizar los datos y obtener una mejor comprensión de la evolución del consumo eléctrico de la vivienda.

        viviendas_sample = df['cups'].drop_duplicates().sample(4, random_state=42)
        plt.figure(figsize=(14, 6))

        for i, vivienda in enumerate(viviendas_sample):
            df_viv = df[df['cups'] == vivienda].sort_values(by=['fecha', 'hora'])
            
            fechas = pd.to_datetime(df_viv['fecha']) + pd.to_timedelta(df_viv['hora'], unit='h')

            plt.plot(fechas, df_viv['consumo_kWh'], 
                    label=f'Vivienda {vivienda[:6]}...', 
                    alpha=0.8)

        plt.title("Consumo horario - Muestra de viviendas")
        plt.xlabel("Fecha y hora")
        plt.ylabel("Consumo (kWh)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_script_dir, '..', 'img_results')
        img_path = f"{output_dir}/Consumo horario - Muestra de viviendas"
        plt.savefig(img_path, bbox_inches='tight')
        #plt.show()


