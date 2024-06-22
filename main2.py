import streamlit as st
import pandas as pd
import numpy as np
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configuración de la página
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("RAG Chatbot con Sentence Bert, Langchain")

# Función para cargar y procesar el CSV
@st.cache_resource
def load_and_process_data(csv_path):
    df = pd.read_csv(csv_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(" ".join(df.values.flatten().astype(str)))
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_texts(texts, embeddings)
    return vectorstore, embeddings

# Cargar y procesar el CSV al inicio
csv_path = "./data_corpus.csv" 
vectorstore, embeddings_model = load_and_process_data(csv_path)

# Inicializar el modelo de lenguaje
api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=api_key)

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

# Función para calcular la similitud coseno
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Inicialización de variables de sesión
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Interfaz de chat
user_question = st.text_input("Haz una pregunta sobre el documento:")
if user_question:
    # Obtener el embedding de la pregunta
    question_embedding = embeddings_model.embed_query(user_question)
    
    # Obtener la respuesta y los documentos recuperados
    response = conversation_chain({"question": user_question})
    answer = response['answer']
    retrieved_docs = response['source_documents']
    
    # Calcular similitudes coseno
    similarities = [cosine_similarity(question_embedding, embeddings_model.embed_query(doc.page_content)) for doc in retrieved_docs]
    
    # Mostrar la respuesta
    st.write(f"AI: {answer}")
    
    # Mostrar detalles técnicos
    with st.expander("Ver detalles técnicos"):
        st.write("Embedding de la pregunta (primeros 10 elementos):")
        st.write(question_embedding[:10])
        
        st.write("Documentos recuperados y similitudes:")
        for doc, sim in zip(retrieved_docs, similarities):
            st.write(f"Documento: {doc.page_content[:100]}...")
            st.write(f"Similitud coseno: {sim:.4f}")
            st.write("---")
    
    st.session_state.chat_history.append((user_question, answer))

# Mostrar el historial de chat
st.write("Historial de Chat:")
for question, answer in st.session_state.chat_history:
    st.write(f"Humano: {question}")
    st.write(f"AI: {answer}")
    st.write("---")