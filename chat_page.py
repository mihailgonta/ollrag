import ollama
import chromadb
import streamlit as st
from scripts.ollama_rag import OllamaRag


def load_collections():
    client = chromadb.PersistentClient(path="data/chroma")
    collections = client.list_collections()
    st.session_state.collections = collections 

if "collections" not in st.session_state:
    st.session_state.collections = []
    load_collections()

st.markdown("<h1 style='text-align: center; color: grey; padding: 2rem 0rem 4rem;'>Ollama RAG 🦙</h1>", unsafe_allow_html=True)

model_col, collection_col, rag_toggle_col = st.columns([4,4,1], vertical_alignment="bottom")

if "rag_on" not in st.session_state:
    st.session_state["rag_on"] = True

with model_col:
    ollama_models = ollama.list()
    models_name = [model.model for model in ollama_models.models]
    selected_model = st.selectbox("Model", models_name)

with rag_toggle_col:
    st.session_state["rag_on"] = st.toggle("RAG")

with collection_col:
    collection_name = st.selectbox("Collection", [col for col in st.session_state.collections], disabled=not st.session_state["rag_on"])

with st.expander("Temperature"):
    temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.8, step=0.1, label_visibility="hidden")
    st.markdown("Increasing the temperature will make the model answer more creatively. (Default: 0.8)")


def call(query):
    with st.chat_message("user"):
        st.write(query)
        
    ollama_rag = OllamaRag(
        embeddings_model='nomic-embed-text',
        ollama_model=selected_model,
        temperature=temperature,
        collection_name=collection_name,
        n_chunks=5
    )
    
    bot = st.chat_message("assistant")
    
    bot.write_stream(ollama_rag.stream_call(query))
    

prompt = st.chat_input("How can I help you?")

if prompt: 
    call(prompt)