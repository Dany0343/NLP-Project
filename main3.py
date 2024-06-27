import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página de Streamlit
st.set_page_config(page_title="Comparador de Textos", layout="wide")
st.title("Comparador de Textos usando Sentence BERT")

# Cargar el Corpus
@st.cache_data
def load_corpus():
    return pd.read_csv('normalized_data_corpus.csv')

corpus_df = load_corpus()

# Crear la carpeta para guardar los archivos .pkl
embeddings_folder = 'embeddings'
if not os.path.exists(embeddings_folder):
    os.makedirs(embeddings_folder)

# Limpieza de datos para manejar valores no textuales en 'title' y 'content_summary'
titles_cleaned = corpus_df['title'].fillna('').apply(str)
content_summaries_cleaned = corpus_df['content_summary'].fillna('').apply(str)
combined_texts_cleaned = [title + " " + summary for title, summary in zip(titles_cleaned, content_summaries_cleaned)]

# Cargar el modelo de Sentence BERT
@st.cache_resource
def load_model():
    return SentenceTransformer('distiluse-base-multilingual-cased-v1')

model = load_model()

# Definición de los archivos pkl
pkl_files = {
    'title': os.path.join(embeddings_folder, 'title_embeddings.pkl'),
    'content': os.path.join(embeddings_folder, 'content_embeddings.pkl'),
    'title_content': os.path.join(embeddings_folder, 'title_content_embeddings.pkl'),
}

# Función para calcular y guardar embeddings
@st.cache_data
def calculate_and_save_embeddings(texts, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = model.encode(texts, show_progress_bar=True)
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings

# Calcular y guardar embeddings
title_embeddings = calculate_and_save_embeddings(titles_cleaned, pkl_files['title'])
content_embeddings = calculate_and_save_embeddings(content_summaries_cleaned, pkl_files['content'])
title_content_embeddings = calculate_and_save_embeddings(combined_texts_cleaned, pkl_files['title_content'])

# Interfaz de usuario con Streamlit
st.sidebar.header("Configuración")
comparison_element = st.sidebar.selectbox(
    "Elemento de comparación",
    ("titulo", "contenido", "titulo+contenido")
)

news_text = st.text_area("Ingrese el texto de la noticia aquí:", height=200)

if st.button("Comparar"):
    if news_text:
        # Generar el embedding para el texto de entrada
        input_embedding = model.encode([news_text])

        # Seleccionar los embeddings correspondientes según la elección del usuario
        if comparison_element == 'titulo':
            corpus_embeddings = title_embeddings
        elif comparison_element == 'contenido':
            corpus_embeddings = content_embeddings
        else:  # 'titulo+contenido'
            corpus_embeddings = title_content_embeddings

        # Calcular la similitud del coseno entre el embedding de entrada y los embeddings del corpus
        similarities = cosine_similarity(input_embedding, corpus_embeddings)[0]

        # Obtener los índices de las 10 noticias más similares
        top_10_indices = np.argsort(similarities)[-10:][::-1]

        # Mostrar las 10 noticias más similares con sus respectivas similitudes de coseno
        st.subheader("Las 10 noticias más similares son:")
        for rank, idx in enumerate(top_10_indices, 1):
            similarity = similarities[idx]
            title = corpus_df.iloc[idx]['title']
            indice = (idx + 2)
            
            st.write(f"{rank}. [{indice}] {title} (Similitud de coseno: {similarity:.8f})")

    else:
        st.warning("Por favor, ingrese un texto para comparar.")

st.sidebar.info("Esta aplicación utiliza Sentence BERT para comparar el texto ingresado con un corpus de noticias y mostrar las 10 más similares.")