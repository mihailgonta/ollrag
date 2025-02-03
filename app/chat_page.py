import os
import uuid
import ollama
import chromadb
import streamlit as st
from scripts.ollama_rag import OllamaAssistant, RAGConfig
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from langfuse import Langfuse
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

langfuse = Langfuse(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host="https://cloud.langfuse.com"
)

if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

if "collections" not in st.session_state:
    st.session_state.collections = []
    
if "use_rag" not in st.session_state:
    st.session_state.use_rag = True

if "feedbacks" not in st.session_state:
    st.session_state.feedbacks = {}

def ollama_inference(prompt, collection, conversation_key):
    # Initialize conversation history if not exists
    if conversation_key not in st.session_state:
        st.session_state[conversation_key] = []
    
    # Display existing chat history first
    display_chat_history(conversation_key)
    
    if prompt:
        # Add user message to history
        st.session_state[conversation_key].append({
            "content": prompt,
            "role": "user"
        })
        
        with st.chat_message("question", avatar="🧑‍🚀"):
            st.write(prompt)

        
        ollama_assistant = OllamaAssistant(
            embeddings=OllamaEmbeddings(model='nomic-embed-text'),
            chat_model=ChatOllama(model=llm_name, temperature=temperature),
            collection_name=collection,
            rag_config=RAGConfig(
                n_chunks=5
            )
        )
        
        response = ollama_assistant.stream_query(
            query=prompt, 
            user_id=st.session_state["user_id"],
            use_rag=st.session_state.use_rag
        )
        
        # Display the new response
        with st.chat_message("response", avatar="🤖"):
            ai_text = st.write_stream(response.content)
            
        # Store the response in chat history
        st.session_state[conversation_key].append({
            "content": ai_text,
            "role": "assistant",
            "trace_id": response.trace_id,
            "feedback_submitted": False,
            "feedback_value": None
        })
        
        st.rerun()


def send_feedback(trace_id, name, value, comment: str = None):
    langfuse.score(
        trace_id=trace_id,
        name=name,
        value=value,
        comment=comment
    )
    st.rerun()


def display_chat_history(chat_history_key):
    for idx, message in enumerate(st.session_state[chat_history_key]):
        role = message["role"]
        
        if role == "user":
            with st.chat_message("user", avatar="🧑‍🚀"):
                st.markdown(message["content"], unsafe_allow_html=True)
        
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"], unsafe_allow_html=True)
            
                feedback_key = f"feedback_{message.get('trace_id', '')}_{idx}_{chat_history_key}"
                comment_key = f"comment_{message.get('trace_id', '')}_{idx}_{chat_history_key}"
                
                if not message.get("feedback_submitted", False):
                    feedback = st.feedback(
                        "thumbs",
                        key=feedback_key
                    )
                    
                    if feedback is not None and message.get("trace_id"):
                        if feedback == 1:
                            message["feedback_submitted"] = True
                            message["feedback_value"] = feedback
                            send_feedback(trace_id=message["trace_id"], name="like", value=feedback)
                        else:
                            # Only show comment box when dislike is selected
                            comment = st.text_area(
                                label="What could be improved?",
                                key=comment_key,
                                placeholder="Please tell us more about why this response wasn't helpful"
                            )
                            
                            if st.button("Submit Feedback", key=f"submit_{feedback_key}"):
                                if comment:  # Only submit if there's a comment
                                    message["feedback_submitted"] = True
                                    message["feedback_value"] = feedback
                                    send_feedback(
                                        trace_id=message["trace_id"],
                                        name="like",
                                        value=feedback,
                                        comment=comment
                                    )
                                    st.success("Thank you for your feedback!")
                                else:
                                    st.warning("Please provide a comment before submitting")


def load_collections():
    """Refresh available Chroma collections"""
    client = chromadb.PersistentClient(path=os.path.join("..", "data", "chroma"))
    st.session_state.collections = client.list_collections()


def get_model_settings():
    model_col, collection_col, rag_toggle_col = st.columns([4, 4, 1], vertical_alignment="bottom")
    with model_col:
        ollama_models = ollama.list()
        selected_model = st.selectbox("Model", [model.model for model in ollama_models.models])

    with rag_toggle_col:
        st.session_state.use_rag = st.toggle("RAG", value=st.session_state.use_rag)

    with collection_col:
        collection_name = st.selectbox(
            "Collection", 
            [col for col in st.session_state.collections], 
            disabled=not st.session_state.use_rag
        )

    with st.expander("Temperature"):
        temperature = st.slider(
            label="Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8, 
            step=0.1, 
            label_visibility="hidden"
        )
        st.markdown("Increasing the temperature will make the model answer more creatively. (Default: 0.8)")
        
    return selected_model, collection_name, temperature

                
load_collections()

st.markdown("<h1 style='text-align: center; color: grey; padding: 2rem 0rem 4rem;'>Ollama RAG 🦙</h1>", unsafe_allow_html=True)

llm_name, collection, temperature = get_model_settings()

if not llm_name: st.stop()

conversation_key = f"model_{llm_name}"

prompt = st.chat_input(f"Ask '{llm_name}' a question ...")

ollama_inference(prompt, collection, conversation_key)