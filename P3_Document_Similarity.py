from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import os.path
import pickle
import stanza
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog

# Cargar el Corpus
corpus_df = pd.read_csv('normalized_data_corpus.csv')

# Crear la carpeta para guardar los archivos .pkl
vectorizers_folder = 'vectorizers'
if not os.path.exists(vectorizers_folder):
    os.makedirs(vectorizers_folder)

# Limpieza de datos para manejar valores no textuales en 'title' y 'content_summary'
titles_cleaned = corpus_df['title'].fillna('').apply(str)
content_summaries_cleaned = corpus_df['content_summary'].fillna('').apply(str)
combined_texts_cleaned = [title + " " + summary for title, summary in zip(titles_cleaned, content_summaries_cleaned)]

# Cargar el modelo de Stanza
nlp = stanza.Pipeline('es')

# Definir lista de palabras a remover
stop_words = {
    'artículos': ['el', 'la', 'lo', 'los', 'las', 'un', 'una', 'unos', 'unas'],
    'preposiciones': ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante', 'en', 'entre', 'hacia',
                      'hasta', 'mediante', 'para', 'por', 'según', 'sin', 'so', 'sobre', 'tras', 'versus', 'vía'],
    'pronombres': ['yo', 'tu', 'el', 'ella', 'eso', 'nosotros', 'ustedes', 'ellos'],
    'conjunciones': ['y', 'o', 'pero', 'porque', 'sino', 'aunque', 'si', 'como', 'cuando', 'mientras', 'más', 'menos',
                     'tan', 'tanto', 'ni', 'que', 'siempre', 'nunca', 'tal', 'cual', 'sino']
}

# Función para normalizar el texto utilizando Stanza y remover palabras específicas
def normalize_and_remove_stopwords(text):
    doc = nlp(text)
    normalized_text = ' '.join([word.lemma for sent in doc.sentences for word in sent.words
                                 if word.lemma.lower() not in stop_words['artículos'] + stop_words['preposiciones'] +
                                 stop_words['pronombres'] + stop_words['conjunciones']])
    return normalized_text

# Definición de los archivos pkl
pkl_files = {
    'title_freq_uni': os.path.join(vectorizers_folder, 'X_vectorizador_title_freq_uni.pkl'),
    'title_freq_bi': os.path.join(vectorizers_folder, 'X_vectorizador_title_freq_bi.pkl'),
    'title_binary_uni': os.path.join(vectorizers_folder, 'X_vectorizador_title_binary_uni.pkl'),
    'title_binary_bi': os.path.join(vectorizers_folder, 'X_vectorizador_title_binary_bi.pkl'),
    'title_tfidf_uni': os.path.join(vectorizers_folder, 'X_vectorizador_title_tfidf_uni.pkl'),
    'title_tfidf_bi': os.path.join(vectorizers_folder, 'X_vectorizador_title_tfidf_bi.pkl'),
    'content_freq_uni': os.path.join(vectorizers_folder, 'X_vectorizador_content_freq_uni.pkl'),
    'content_freq_bi': os.path.join(vectorizers_folder, 'X_vectorizador_content_freq_bi.pkl'),
    'content_binary_uni': os.path.join(vectorizers_folder, 'X_vectorizador_content_binary_uni.pkl'),
    'content_binary_bi': os.path.join(vectorizers_folder, 'X_vectorizador_content_binary_bi.pkl'),
    'content_tfidf_uni': os.path.join(vectorizers_folder, 'X_vectorizador_content_tfidf_uni.pkl'),
    'content_tfidf_bi': os.path.join(vectorizers_folder, 'X_vectorizador_content_tfidf_bi.pkl'),
    'title_content_freq_uni': os.path.join(vectorizers_folder, 'X_vectorizador_title_content_freq_uni.pkl'),
    'title_content_freq_bi': os.path.join(vectorizers_folder, 'X_vectorizador_title_content_freq_bi.pkl'),
    'title_content_binary_uni': os.path.join(vectorizers_folder, 'X_vectorizador_title_content_binary_uni.pkl'),
    'title_content_binary_bi': os.path.join(vectorizers_folder, 'X_vectorizador_title_content_binary_bi.pkl'),
    'title_content_tfidf_uni': os.path.join(vectorizers_folder, 'X_vectorizador_title_content_tfidf_uni.pkl'),
    'title_content_tfidf_bi': os.path.join(vectorizers_folder, 'X_vectorizador_title_content_tfidf_bi.pkl'),
}

# Definición de las configuraciones de los vectorizadores
vectorizers_config = {
    'title_freq_uni': (CountVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), titles_cleaned),
    'title_freq_bi': (CountVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), titles_cleaned),
    'title_binary_uni': (CountVectorizer(ngram_range=(1, 1), binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), titles_cleaned),
    'title_binary_bi': (CountVectorizer(ngram_range=(2, 2), binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), titles_cleaned),
    'title_tfidf_uni': (TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), titles_cleaned),
    'title_tfidf_bi': (TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), titles_cleaned),
    'content_freq_uni': (CountVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), content_summaries_cleaned),
    'content_freq_bi': (CountVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), content_summaries_cleaned),
    'content_binary_uni': (CountVectorizer(ngram_range=(1, 1), binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), content_summaries_cleaned),
    'content_binary_bi': (CountVectorizer(ngram_range=(2, 2), binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), content_summaries_cleaned),
    'content_tfidf_uni': (TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), content_summaries_cleaned),
    'content_tfidf_bi': (TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), content_summaries_cleaned),
    'title_content_freq_uni': (CountVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), combined_texts_cleaned),
    'title_content_freq_bi': (CountVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), combined_texts_cleaned),
    'title_content_binary_uni': (CountVectorizer(ngram_range=(1, 1), binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), combined_texts_cleaned),
    'title_content_binary_bi': (CountVectorizer(ngram_range=(2, 2), binary=True, token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), combined_texts_cleaned),
    'title_content_tfidf_uni': (TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), combined_texts_cleaned),
    'title_content_tfidf_bi': (TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), combined_texts_cleaned),
}

# Loop para cargar o crear los archivos pkl
for vector_name, pkl_file_name in pkl_files.items():
    if os.path.exists(pkl_file_name):
        with open(pkl_file_name, 'rb') as vector_file:
            X_vector = pickle.load(vector_file)
    else:
        vector_file = open(pkl_file_name, 'wb')
        vectorizer, corpus = vectorizers_config[vector_name]
        X_vector = vectorizer.fit_transform(corpus)
        pickle.dump(X_vector, vector_file)
        vector_file.close()    

# Crear la ventana principal de Tkinter
root = tk.Tk()

while True:
    # Mostrar el explorador de archivos y obtener la ruta del archivo seleccionado
    file_path = filedialog.askopenfilename(title="Seleccione el archivo de texto", filetypes=[("Archivos de texto", "*.txt")])

    # Leer el contenido del archivo seleccionado
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            news_text = file.read()
        print("Contenido del archivo seleccionado:")
        print(news_text)
    else:
        print("No se seleccionó ningún archivo.")
        break

    # Normalizar la noticia 
    test = normalize_and_remove_stopwords(news_text)

    # Solicitar input del usuario para comparación, extracción y tipo de representación
    comparison_element = input("Ingrese el elemento de comparación (titulo, contenido, titulo+contenido): ")
    extraction_values = input("Ingrese los valores a extraer (unigramas, bigramas): ")
    representation_type = input("Ingrese el tipo de representación (binarizada, frecuencia, tfidf): ")

    # Generar la representación del espacio vectorial
    if comparison_element == 'titulo':
        if extraction_values == 'unigramas':
            if representation_type == 'binarizada':
                vector_name = 'title_binary_uni'
            elif representation_type == 'frecuencia':
                vector_name = 'title_freq_uni'
            elif representation_type == 'tfidf':
                vector_name = 'title_tfidf_uni'
        elif extraction_values == 'bigramas':
            if representation_type == 'binarizada':
                vector_name = 'title_binary_bi'
            elif representation_type == 'frecuencia':
                vector_name = 'title_freq_bi'
            elif representation_type == 'tfidf':
                vector_name = 'title_tfidf_bi'
    elif comparison_element == 'contenido':
        if extraction_values == 'unigramas':
            if representation_type == 'binarizada':
                vector_name = 'content_binary_uni'
            elif representation_type == 'frecuencia':
                vector_name = 'content_freq_uni'
            elif representation_type == 'tfidf':
                vector_name = 'content_tfidf_uni'
        elif extraction_values == 'bigramas':
            if representation_type == 'binarizada':
                vector_name = 'content_binary_bi'
            elif representation_type == 'frecuencia':
                vector_name = 'content_freq_bi'
            elif representation_type == 'tfidf':
                vector_name = 'content_tfidf_bi'
    elif comparison_element == 'titulo+contenido':
        if extraction_values == 'unigramas':
            if representation_type == 'binarizada':
                vector_name = 'title_content_binary_uni'
            elif representation_type == 'frecuencia':
                vector_name = 'title_content_freq_uni'
            elif representation_type == 'tfidf':
                vector_name = 'title_content_tfidf_uni'
        elif extraction_values == 'bigramas':
            if representation_type == 'binarizada':
                vector_name = 'title_content_binary_bi'
            elif representation_type == 'frecuencia':
                vector_name = 'title_content_freq_bi'
            elif representation_type == 'tfidf':
                vector_name = 'title_content_tfidf_bi'

    # Cargar el vectorizador correspondiente
    with open(pkl_files[vector_name], 'rb') as vector_file:
        X_vector = pickle.load(vector_file)

    # Cargar el vectorizador correspondiente y ajustarlo
    vectorizer, corpus = vectorizers_config[vector_name]
    vectorizer.fit(corpus)

    # Transformar la noticia normalizada
    Y_vector = vectorizer.transform([test])

    # Calcular la similitud del coseno entre Y_vector y cada vector en X_vector
    similarities = cosine_similarity(Y_vector, X_vector)

    # Obtener los índices de las 10 noticias más similares
    top_10_indices = np.argsort(similarities[0])[-10:][::-1]

    # Imprimir las 10 noticias más similares con sus respectivas similitudes de coseno
    print("Las 10 noticias más similares son:")
    for rank, idx in enumerate(top_10_indices, 1):
        similarity = similarities[0, idx]  # Obtener la similitud de coseno correspondiente
        title = corpus_df.iloc[idx]['title']  # Obtener el título de la noticia correspondiente
        indice = (idx+2)
        
        print(f"{rank}   [{indice}] \t  {title} \t\t (Similitud de coseno: {similarity:.8f})")

    # Preguntar al usuario si desea realizar otra comparación
    respuesta = input("¿Desea realizar otra comparación? (s/n): ")
    if respuesta.lower() != 's':
        break  
