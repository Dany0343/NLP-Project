import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import stanza

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(page_title="RAG Chatbot y Comparación TF-IDF", page_icon="🤖", layout="wide")
st.title("RAG Chatbot con Sentence BERT, Langchain 🦜, ChromaDB y Comparación TF-IDF")

# Función para cargar y procesar el CSV
@st.cache_resource
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)
    texts = text_splitter.split_text(" ".join(df.values.flatten().astype(str)))
    
    embeddings = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")
    
    vectorstore = Chroma.from_texts(texts, embeddings)
    return vectorstore, embeddings

# Cargar y procesar el CSV al inicio
csv_path = "./data_corpus.csv" 
vectorstore, embeddings_model = load_and_process_data(csv_path)

# Inicializar el modelo de lenguaje
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo", api_key=api_key)

# Configurar la memoria
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Configurar el prompt
prompt_template = """Use el siguiente contexto para contestar la respuesta del final, si no sabes la respuesta di que no sabes, pero usa tu conocimiento sin inventar la respuesta.

{context}

Pregunta: {question}
Historial relevante de la conversación: {chat_history}
Respuesta:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question", "chat_history"]
)

# Inicializar la cadena de conversación
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Inicialización de variables de sesión
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Crear dos columnas
col1, col2 = st.columns([1, 2])

# Columna izquierda para el historial del chat
with col1:
    st.subheader("Historial de Chat")
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
        st.write("---")

# Columna derecha para la interfaz de chat y detalles técnicos
with col2:
    # Interfaz de chat
    user_question = st.text_input("Haz una pregunta sobre el documento:")
    if user_question:
        # Obtener el embedding de la pregunta
        question_embedding = np.array(embeddings_model.embed_query(user_question)).reshape(1, -1)
        
        # Obtener la respuesta y los documentos recuperados
        response = conversation_chain({"question": user_question})
        answer = response['answer']
        retrieved_docs = response['source_documents']
        
        # Calcular similitudes coseno
        similarities = [cosine_similarity(question_embedding, np.array(embeddings_model.embed_query(doc.page_content)).reshape(1, -1)).flatten()[0] for doc in retrieved_docs]
        
        # Mostrar la respuesta
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            st.write(f"AI: {answer}")
        
        # Mostrar detalles técnicos
        with st.expander("Ver detalles técnicos"):
            st.write("Embedding de la pregunta (primeros 10 elementos):")
            st.write(question_embedding.flatten()[:10])
            
            st.write("Documentos recuperados y similitudes:")
            for doc, sim in zip(retrieved_docs, similarities):
                st.write(f"Documento: {doc.page_content[:100]}...")
                st.write(f"Similitud coseno: {sim:.4f}")
                st.write("---")
            
            # Mostrar gráfica de similitudes
            fig, ax = plt.subplots()
            doc_labels = [f"Doc {i+1}" for i in range(len(retrieved_docs))]
            ax.bar(doc_labels, similarities)
            ax.set_xlabel("Documentos")
            ax.set_ylabel("Similitud Coseno")
            ax.set_title("Similitudes Coseno de Documentos Recuperados")
            st.pyplot(fig)
        
        st.session_state.chat_history.append((user_question, answer))

        # Procesamiento de la parte de TF-IDF
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
            'content_freq_uni': (CountVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), content_summaries_cleaned),
            'content_freq_bi': (CountVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), content_summaries_cleaned),
            'content_binary_uni': (CountVectorizer(ngram_range=(1, 1), binary=True, token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), content_summaries_cleaned),
            'content_binary_bi': (CountVectorizer(ngram_range=(2, 2), binary=True, token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), content_summaries_cleaned),
            'content_tfidf_uni': (TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), content_summaries_cleaned),
            'content_tfidf_bi': (TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), content_summaries_cleaned),
            'title_content_freq_uni': (CountVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), combined_texts_cleaned),
            'title_content_freq_bi': (CountVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), combined_texts_cleaned),
            'title_content_binary_uni': (CountVectorizer(ngram_range=(1, 1), binary=True, token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), combined_texts_cleaned),
            'title_content_binary_bi': (CountVectorizer(ngram_range=(2, 2), binary=True, token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), combined_texts_cleaned),
            'title_content_tfidf_uni': (TfidfVectorizer(ngram_range=(1, 1), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), combined_texts_cleaned),
            'title_content_tfidf_bi': (TfidfVectorizer(ngram_range=(2, 2), token_pattern=r'(?u)\w+|\w+\\n|\.|\¿|\?'), combined_texts_cleaned),
        }

        # Loop para cargar o crear los archivos pkl
        def load_or_create_vectorizers():
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

        load_or_create_vectorizers()

        # Normalizar la noticia 
        test = normalize_and_remove_stopwords(user_question)

        # Configuración de comparación
        comparison_element = st.selectbox('Selecciona el elemento de comparación', ['titulo', 'contenido', 'titulo+contenido'])
        extraction_values = st.selectbox('Selecciona los valores a extraer', ['unigramas', 'bigramas'])
        representation_type = st.selectbox('Selecciona el tipo de representación', ['binarizada', 'frecuencia', 'tfidf'])

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
        st.subheader("Las 10 noticias más similares son:")
        for rank, idx in enumerate(top_10_indices, 1):
            similarity = similarities[0, idx]  # Obtener la similitud de coseno correspondiente
            title = corpus_df.iloc[idx]['title']  # Obtener el título de la noticia correspondiente
            st.write(f"{rank}. {title} (Similitud de coseno: {similarity:.8f})")
