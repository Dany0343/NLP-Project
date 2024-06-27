from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os
import pickle
import stanza
from sklearn.metrics.pairwise import cosine_similarity

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

# Definición del archivo pkl y configuración del vectorizador
pkl_file_name = os.path.join(vectorizers_folder, 'X_vectorizador_title_content_tfidf_uni.pkl')
vectorizer_config = (TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\n|\.|\¿|\?'), combined_texts_cleaned)

# Cargar o crear el archivo pkl
if os.path.exists(pkl_file_name):
    with open(pkl_file_name, 'rb') as vector_file:
        X_vector = pickle.load(vector_file)
else:
    vector_file = open(pkl_file_name, 'wb')
    vectorizer, corpus = vectorizer_config
    X_vector = vectorizer.fit_transform(corpus)
    pickle.dump(X_vector, vector_file)
    vector_file.close()

while True:
    # Solicitar entrada del usuario
    news_text = input("Ingrese el contenido de la noticia: ")
    
    # Normalizar la noticia 
    test = normalize_and_remove_stopwords(news_text)

    # Cargar el vectorizador correspondiente y ajustarlo
    vectorizer, corpus = vectorizer_config
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
