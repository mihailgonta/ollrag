import os
import time
import chromadb
import streamlit as st
import pandas as pd
from scripts.ollama_data import OllamaDb
from pathlib import Path

client = chromadb.PersistentClient(path=os.path.join("..", "data", "chroma"))


def load_collections():
    st.session_state.collections = []
    collection_names = list(client.list_collections())

    for collection_name in collection_names:
        collection = client.get_collection(name=collection_name)
        st.session_state.collections.append(collection)


def display_collections_table_advanced():
    # Create checkbox state if it doesn't exist
    if "selected_collections" not in st.session_state:
        st.session_state.selected_collections = set()

    # Prepare data
    data = []
    for collection in st.session_state.collections:
        metadata = collection.metadata or {}
        creation_date = "unknown"

        if metadata and "date" in metadata:
            date = metadata.get("date")
            creation_date = date.strftime("%d.%m.%y") if date else "unknown"

        data.append(
            {
                "Select": False,  # Checkbox column
                "Name": collection.name,
                "Count": collection.count(),
                "Created": creation_date,
            }
        )

    if not data:
        st.info("No collections found")
        return

    df = pd.DataFrame(data)

    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        search = st.text_input("Search collections", "")
    with col2:
        sort_by = st.selectbox("Sort by", ["Name", "Count", "Created"])

    # Apply filters
    if search:
        df = df[df["Name"].str.contains(search, case=False)]

    # Sort
    if sort_by in ["Name", "Count", "Created"]:
        df = df.sort_values(by=sort_by)

    # Display table with checkboxes
    edited_df = st.data_editor(
        df,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select", help="Select collection", default=False
            ),
            "Name": st.column_config.TextColumn(
                "Collection Name", help="Name of the collection"
            ),
            "Count": st.column_config.NumberColumn(
                "Documents", help="Number of documents in collection"
            ),
            "Created": st.column_config.TextColumn(
                "Creation Date", help="When the collection was created"
            ),
        },
        hide_index=True,
        use_container_width=True,
        disabled=["Name", "Count", "Created"],
    )

    # Update selected collections based on checkboxes
    selected_collections = edited_df[edited_df["Select"]]["Name"].tolist()
    st.session_state.selected_collections = set(selected_collections)

    # Display action buttons for selected collections
    if st.session_state.selected_collections:
        st.write(f"Selected: {len(st.session_state.selected_collections)} collections")

        col1, col2 = st.columns([4, 10])
        with col1:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state.show_delete_confirm = True

        # Show delete confirmation
        if st.session_state.get("show_delete_confirm", False):
            with st.container():
                st.warning(
                    f"Are you sure you want to delete {len(st.session_state.selected_collections)} collections?"
                )
                conf_col1, conf_col2 = st.columns(2)
                with conf_col1:
                    if st.button("Yes, delete"):
                        for collection_name in st.session_state.selected_collections:
                            try:
                                client.delete_collection(collection_name)
                            except Exception as e:
                                st.error(f"Error deleting {collection_name}: {str(e)}")
                        st.success("Selected collections deleted successfully")
                        st.session_state.selected_collections = set()
                        st.session_state.show_delete_confirm = False
                        load_collections()
                        st.rerun()
                with conf_col2:
                    if st.button("Cancel"):
                        st.session_state.show_delete_confirm = False
                        st.rerun()


st.title("Collections Manager")

if "collections" not in st.session_state:
    load_collections()

display_collections_table_advanced()
