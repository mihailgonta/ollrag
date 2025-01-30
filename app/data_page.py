import os
import chromadb
import streamlit as st
from scripts.ollama_data import OllamaDb

client = chromadb.PersistentClient(path="../data/chroma")

def load_collections():
    collections = client.list_collections()
    st.session_state.collections = collections 


def create_collection(collection_name: str, max_chunks:int = 400, min_chunks: int = 50):
    ollama_db = OllamaDb('nomic-embed-text')
    
    file_paths = [
        os.path.join(folder_path, file_name)
        for key, value in st.session_state.items()
        if key not in ["temp_folder_paths"] and value and "_" in key 
        for folder_path, file_name in [key.rsplit("_", 1)]
    ]
    
    documents = ollama_db.load_documents(file_paths)
    chunks = ollama_db.chunk_documents(documents, max_chunks=max_chunks, min_chunks=min_chunks)
    ollama_db.create_collection(chunks, collection_name)
    
    load_collections()


if "collections" not in st.session_state:
    st.session_state.collections = []
    load_collections()

if "temp_folder_paths" not in st.session_state:
    st.session_state.temp_folder_paths = []
if "create_collection_success" not in st.session_state:
    st.session_state.create_collection_success = False

@st.dialog("Add a new collection", width="large")
def add_collection(): 
    folder_path = ""
    with st.container(border=True):
        input_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
        
        with input_col:
            folder_path = st.text_input(label="Directory Path", placeholder="path/to/your/files")
        
        with button_col:
            if st.button("Add folder", icon="📂"):
                if os.path.exists(folder_path):
                    if folder_path not in st.session_state.temp_folder_paths:
                        st.session_state.temp_folder_paths.append(folder_path)
                    else:
                        st.toast("Folder already added.")
                else:
                    st.toast("Path not found.")
    
    with st.container(border=False):
        for folder_path in st.session_state.temp_folder_paths:
            with st.expander(folder_path):
                files_list = [f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.txt', '.md'))]
                
                if not files_list:
                    st.write("No PDF or TXT files found in this folder.")
                else:
                    with st.container(border=True, height=400):
                        all_selected = st.checkbox("🗃️ Include all", key=f"{folder_path}_all")
                        
                        for file_name in files_list:
                            file_key = f"{folder_path}_{file_name}"
                            st.checkbox(f"📄 {file_name}", value=all_selected, key=file_key)
                        
    with st.container(border=True):
        collection_name_col, create_button_col = st.columns([7, 1], vertical_alignment="bottom")

    with collection_name_col:
        new_collection_name = st.text_input(label="Collection name", placeholder="myCollection")

    with create_button_col:
        if st.button("Create", type="primary"):
            if new_collection_name not in st.session_state.collections:
                create_collection(new_collection_name)
                st.rerun()
            else:
                st.toast("Collection already exist.")

if not st.session_state.get("add_collection_open", False):
    st.session_state.temp_folder_paths = []


st.markdown("<h1 style='text-align: center; color: grey; padding: 2rem 0rem 4rem;'>Ollama RAG 🦙</h1>", unsafe_allow_html=True)


@st.dialog("Delete collection")
def delete_collection(idx, collection_name):
    st.warning(f"Are you sure?", icon="⚠️")
    
    if st.button("Yes"):
        client.delete_collection(collection_name)
        st.session_state['collections'].pop(idx)
        st.rerun()


def display_collection(collection_name, idx):
    tile = col.container(border=True)
    with tile:
        with st.container(border=False):
            title_col, edit_button_col = st.columns([2, 1], vertical_alignment="center")
            with title_col:
                st.markdown(f"""
                    <p style='font-size: 1.5rem; w
                    hite-space: nowrap; 
                    overflow: hidden; 
                    text-overflow: ellipsis;' 
                    title='{collection_name}'>
                        {collection_name}
                    </p>
                """,unsafe_allow_html=True)
            
            with edit_button_col:
                with st.popover(label="⋮", use_container_width=True):
                    if st.button("Delete", icon="🗑️", key=f"{collection_name}_delete_button"):
                        delete_collection(idx, collection_name)
                    
                    if st.button("Update", icon="📝", key=f"{collection_name}_edit_button"):
                        pass
        
            st.write("<hr style='margin: 0.6rem 0rem'>", unsafe_allow_html=True)
        
            with st.container(border=False):
                date_col_1, date_col_2 = st.columns([2, 1], vertical_alignment="center")
                
                with date_col_1:
                    st.markdown(f"<p style='font-size: 1rem; opacity: 0.5;'>Date modified:</p>", unsafe_allow_html=True)
                with date_col_2:    
                    st.markdown(f"<p style='font-size: 1rem; opacity: 0.5;'>12.12.24</p>", unsafe_allow_html=True)


with st.container():
    with st.container(border=True):
        title_col, button_col = st.columns([9, 1], vertical_alignment="center")
        
        with title_col:
            st.markdown(f"<p style='font-size: 1.5rem'>Collections</p>", unsafe_allow_html=True)
        with button_col:
            if st.button("Add"):
                add_collection()
                
    columns_per_row = 3
    for idx, collection_name in enumerate(st.session_state.collections):
        if idx % columns_per_row == 0:
            row = st.columns(columns_per_row)  # Start a new row
        col = row[idx % columns_per_row]
        with col:
            display_collection(collection_name, idx)