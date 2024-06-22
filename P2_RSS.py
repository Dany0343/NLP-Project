import feedparser
import csv
import stanza
from datetime import datetime
import os.path
import subprocess

# Descargar los modelos de Stanza 
# stanza.download('es')

# Cargar el modelo de Stanza
# nlp = stanza.Pipeline('es')

# # Definir lista de palabras a remover
# stop_words = {
#     'artículos': ['el', 'la', 'lo', 'los', 'las', 'un', 'una', 'unos', 'unas'],
#     'preposiciones': ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia',
#                       'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía'],
#     'pronombres': ['yo', 'tu', 'el', 'ella', 'eso', 'nosotros', 'ustedes', 'ellos'],
#     'conjunciones': ['y', 'o', 'pero', 'porque', 'sino', 'aunque', 'si', 'como', 'cuando', 'mientras', 'más', 'menos',
#                      'tan', 'tanto', 'ni', 'que', 'siempre', 'nunca', 'tal', 'cual', 'sino']
# }

# # Función para normalizar el texto utilizando Stanza y remover palabras específicas
# def normalize_and_remove_stopwords(text):
#     doc = nlp(text)
#     normalized_text = ' '.join([word.lemma for sent in doc.sentences for word in sent.words
#                                  if word.lemma.lower() not in stop_words['artículos'] + stop_words['preposiciones'] +
#                                  stop_words['pronombres'] + stop_words['conjunciones']])
#     return normalized_text

# Función para formatear la fecha
def format_date(date_string):
    # Parsear la fecha
    date_obj = datetime.strptime(date_string, '%a, %d %b %Y %H:%M:%S %Z')
    # Formatear la fecha en el formato deseado
    formatted_date = date_obj.strftime('%d/%m/%Y')
    return formatted_date

# Definir URLs de los feeds RSS para cada categoría y fuente
rss_urls = {
    'Deportes': {
        'La Jornada': 'https://www.jornada.com.mx/rss/deportes.xml?v=1',
        'Expansión': 'https://expansion.mx/rss/deportes'
    },
    'Economía': {
        'La Jornada': 'https://www.jornada.com.mx/rss/economia.xml?v=1',
        'Expansión': 'https://expansion.mx/rss/economia'
    },
    'Ciencia y tecnología': {
        'La Jornada': 'https://www.jornada.com.mx/rss/ciencias.xml?v=1',
        'Expansión': 'https://expansion.mx/rss/tecnologia'
    },
    'Cultura': {
        'La Jornada': 'https://www.jornada.com.mx/rss/cultura.xml?v=1',
        'Expansión': 'https://expansion.mx/rss/cultura'
    },
    'Sociedad y Justicia': {
        'La Jornada': 'https://www.jornada.com.mx/rss/sociedad.xml?v=1',
    },
    'Mundo': {
        'La Jornada': 'https://www.jornada.com.mx/rss/mundo.xml?v=1',
        'Expansión': 'https://expansion.mx/rss/mundo'
    },
    'Tendencias': {
        'Expansión': 'https://expansion.mx/rss/tendencias'
    },
    'Emprendimiento': {
        'Expansión': 'https://expansion.mx/rss/emprendedores'
    },
    'Opinión': {
        'La Jornada': 'https://www.jornada.com.mx/rss/opinion.xml?v=1',
        'Expansión': 'https://expansion.mx/rss/opinion'
    },
    'Espectáculos': {
        'La Jornada': 'https://www.jornada.com.mx/rss/espectaculos.xml?v=1',
    },
    'Gastronomía': {
        'La Jornada': 'https://www.jornada.com.mx/rss/gastronomia.xml?v=1',
    },
    'Nacional': {
        'La Jornada': [
            'https://www.jornada.com.mx/rss/capital.xml?v=1',
            'https://www.jornada.com.mx/rss/estados.xml?v=1'
        ]
    },
}

# Nombre de los archivos CSV
raw_data_file = 'raw_data_corpus.csv'
normalized_data_file = 'normalized_data_corpus.csv'

# Verificar si los archivos CSV existen
raw_data_exists = os.path.exists(raw_data_file)
normalized_data_exists = os.path.exists(normalized_data_file)

# Preparar el archivo CSV de datos crudos
with open(raw_data_file, mode='a' if raw_data_exists else 'w', newline='', encoding='utf-8-sig') as csv_file:
    fieldnames = ['Category', 'Source', 'Title', 'Content Summary', 'URL', 'Date of Publication']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not raw_data_exists:
        writer.writeheader()

    # Iterar a través de cada categoría y fuente, y recolectar noticias
    for category, sources in rss_urls.items():
        for source, url in sources.items():
            # Manejar caso especial para la categoría 'Nacional' que contiene una lista de URLs
            if isinstance(url, list):
                for u in url:
                    feed = feedparser.parse(u)
                    # Iterar a través de las entradas y escribir en el archivo CSV
                    for entry in feed.entries:
                        formatted_date = format_date(entry.published)
                        writer.writerow({
                            'Category': category,
                            'Source': source,
                            'Title': entry.title,
                            'Content Summary': entry.summary,
                            'URL': entry.link,
                            'Date of Publication': formatted_date
                        })
            else:
                feed = feedparser.parse(url)
                # Iterar a través de las entradas y escribir en el archivo CSV
                for entry in feed.entries:
                    formatted_date = format_date(entry.published)
                    writer.writerow({
                        'Category': category,
                        'Source': source,
                        'Title': entry.title,
                        'Content Summary': entry.summary,
                        'URL': entry.link,
                        'Date of Publication': formatted_date
                    })

# # Preparar el archivo CSV de datos normalizados
# with open(normalized_data_file, mode='a' if normalized_data_exists else 'w', newline='', encoding='utf-8-sig') as csv_file:
#     fieldnames = ['Category', 'Source', 'Title', 'Content Summary', 'URL', 'Date of Publication']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#     if not normalized_data_exists:
#         writer.writeheader()
    
#     # Iterar a través de cada categoría y fuente, y recolectar noticias
#     for category, sources in rss_urls.items():
#         for source, url in sources.items():
#             feed = feedparser.parse(url)
            
#             # Iterar a través de las entradas y escribir en el archivo CSV
#             for entry in feed.entries:
#                 # Normalizar los campos de texto usando Stanza
#                 normalized_title = normalize_and_remove_stopwords(entry.title)
#                 normalized_summary = normalize_and_remove_stopwords(entry.summary)
#                 formatted_date = format_date(entry.published)

                
#                 writer.writerow({
#                     'Category': category,
#                     'Source': source,
#                     'Title': normalized_title,
#                     'Content Summary': normalized_summary,
#                     'URL': entry.link,
#                     'Date of Publication': formatted_date
#                 })

# Llamar al script Filtrado.py
subprocess.call(['python', 'Filtrado.py'])

print("Proceso completado exitosamente.")
