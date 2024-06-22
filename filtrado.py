import csv

def eliminar_repetidos(archivo_entrada, columna, archivo_salida):
    # Crear un conjunto para almacenar los valores únicos de la columna
    valores_unicos = set()

    # Leer el archivo CSV de entrada y eliminar filas completas con elementos repetidos en la columna especificada
    with open(archivo_entrada, 'r', newline='', encoding='utf-8-sig') as archivo_entrada, \
            open(archivo_salida, 'w', newline='', encoding='utf-8-sig') as archivo_salida:
        lector_csv = csv.DictReader(archivo_entrada)
        escritor_csv = csv.DictWriter(archivo_salida, fieldnames=lector_csv.fieldnames)
        escritor_csv.writeheader()

        for fila in lector_csv:
            if fila[columna] not in valores_unicos:
                valores_unicos.add(fila[columna])
                escritor_csv.writerow(fila)

# Especifica el nombre del archivo CSV de entrada, la columna a procesar y el nombre del archivo de salida
archivo_entrada_raw = 'raw_data_corpus.csv'
# archivo_entrada_normalized = 'normalized_data_corpus.csv'
columna = 'Title'
archivo_salida_raw = 'raw_data_corpus_filtrado.csv'
# archivo_salida_normalized = 'normalized_data_corpus_filtrado.csv'

# Llamar a la función para eliminar filas completas con elementos repetidos en la columna especificada
eliminar_repetidos(archivo_entrada_raw, columna, archivo_salida_raw)
# eliminar_repetidos(archivo_entrada_normalized, columna, archivo_salida_normalized)

print("Proceso de filtrado realizado correctamente")
