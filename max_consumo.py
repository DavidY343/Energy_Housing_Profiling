import csv

def encontrar_max_consumo(archivo_csv):
    max_consumo = 0
    fecha_max = ""
    hora_max = ""
    
    with open(archivo_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        
        for fila in reader:
            # Convertir el consumo a float (reemplazando comas por puntos para decimales)
            consumo = float(fila['consumo_kWh'].replace(',', '.'))
            
            if consumo > max_consumo:
                max_consumo = consumo
                fecha_max = fila['fecha']
                hora_max = fila['hora']
    
    return max_consumo, fecha_max, hora_max

# Nombre del archivo CSV (ajusta según tu archivo)

archivo_csv1 = 'data_raw/11c65598e4aa Consumo 30-08-2021_30-08-2023.csv'
archivo_csv2 = 'data_raw/ad7b0d773a4a Consumo 29-08-2021_29-08-2023.csv'
archivo_csv3 = 'data_raw/31a15c8a5985 Consumo 07-08-2021_07-08-2023.csv'

# Ejecutar la función
max_consumo, fecha, hora = encontrar_max_consumo(archivo_csv1)

# Mostrar resultados
print(f"El máximo consumo fue de {max_consumo} kWh")
print(f"Fecha: {fecha}")
print(f"Hora: {hora}")

# Ejecutar la función
max_consumo, fecha, hora = encontrar_max_consumo(archivo_csv2)

# Mostrar resultados
print(f"El máximo consumo fue de {max_consumo} kWh")
print(f"Fecha: {fecha}")
print(f"Hora: {hora}")

# Ejecutar la función
max_consumo, fecha, hora = encontrar_max_consumo(archivo_csv3)

# Mostrar resultados
print(f"El máximo consumo fue de {max_consumo} kWh")
print(f"Fecha: {fecha}")
print(f"Hora: {hora}")