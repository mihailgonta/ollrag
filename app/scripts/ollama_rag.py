import os
# import torch
import openai

import chromadb
from langchain_chroma import Chroma

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document

from .utils import openai_token_count

from sentence_transformers import CrossEncoder
from FlagEmbedding import LayerWiseFlagLLMReranker

from langfuse.decorators import observe, langfuse_context

from huggingface_hub import login

from dotenv import load_dotenv, find_dotenv


class OllamaRag:
    def __init__(self, embeddings_model: str, ollama_model: str, temperature: float, collection_name: str, n_chunks: int = 5):
        _ = load_dotenv(find_dotenv())
        
        openai.api_key = os.environ['OPENAI_API_KEY']
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        login(token=hf_token)
        
        self.temperature = temperature
        self.collection_name = collection_name
        self.n_chunks = n_chunks
        self.embeddings_model = OllamaEmbeddings(model=embeddings_model)
        self.ollama_model = ChatOllama(model=ollama_model, temperature=temperature, num_ctx=4096)
        chroma_path = os.path.join("..", "data", "chroma")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        # self.cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')
        # self.flag_reranker = LayerWiseFlagLLMReranker("BAAI/bge-reranker-v2-minicpm-layerwise")
        
    
    def __get_chat_prompt(self):
        HUMAN_MESSAGE_TEMPLATE = """
        You are an agent who provide answers strictly based on the provided context below:
        {context}
        --
        Answer the following question based on the above context:{question}\n
        Your responses should be concise, straight to the point, avoid filling words.\n
        If possible include price. If you find a list that responds to the question, include it in response.\n
        Be careful when identifying plans, e.g commercial plans and noncommercial plans.
        If the question cannot be answered using the given context, respond with: 'I don't know.'\n
        If no context is provided, also respond with: 'I don't know.'\n
        Do not add information or make assumptions beyond the given context.\n
        """
        
        return ChatPromptTemplate.from_template(HUMAN_MESSAGE_TEMPLATE)

    
    def __augment_multiple_query(self, query):
        PROMPT_TEMPLATE = """
        You are a general purpose research assistant.
        Do not number the questions.
        Your users are asking questions on different topics.
        Suggest up to five additional related questions to help them find the information they need, for the provided question.
        Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
        Make sure they are complete questions, and that they are related to the original question.
        Output one question per line. Do not number the questions.\n
        - -
        User question: {question}
        """
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(question=query)
        
        response = self.ollama_model.invoke(prompt).content
        
        questions = response.split("\n")
        
        return questions
    
    
    def __remove_duplicates(self, documents: list[Document]):
        return list({doc.page_content: doc for doc in documents}.values())
    
    
    def __query(self, embedded_queries) -> list[Document]:
        collection = Chroma(collection_name=self.collection_name, persist_directory=os.path.join("..", "data", "chroma"))
        
        results = []
        
        for embedding in embedded_queries:
            results.extend(collection.similarity_search_by_vector(embedding=embedding, k=self.n_chunks))
        
        unique_documents = self.__remove_duplicates(documents=results)
        
        return unique_documents


    def __embed_query(self, query: str, augment_query: bool):
        queries = [query]
        
        if augment_query:
            # Augmenting initial query
            augmented_queries = self.__augment_multiple_query(query)
            queries.extend(augmented_queries)
        
        # Embedding all queries in a single list 
        embedded_query = [self.embeddings_model.embed_query(query) for query in queries]

        return embedded_query
    

    def __rerank(self, query: str, documents: list[Document], top_k: int):
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        # scores = self.flag_reranker.compute_score(pairs, cutoff_layers=[28])
        
        doc_scores = list(zip(documents, scores))
        
        sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, score in sorted_doc_scores]
        
        l = len(documents)
        n = max(min(l, top_k), 0)
        
        return reranked_docs[:n]
    
    
    def __build_context(self, documents: list[Document]):
        context = "\n\n - -\n\n".join(doc.page_content for doc in documents)
        
        return context
    
    
    def __rag(self, query: str, augment_query: bool, rerank: bool, top_k: int = 3):
        embedded_query = self.__embed_query(query, augment_query)
        
        documents = self.__query(embedded_query)

        if rerank:
            documents = self.__rerank(query, documents, top_k)
        
        context = self.__build_context(documents)
        
        prompt = self.__get_chat_prompt()
        
        return prompt, context, documents

    @observe()
    def call(self, query: str, user_id: str = None, augment_query: bool = True, rerank: bool = True, top_k: int = 3):
        langfuse_handler = langfuse_context.get_current_langchain_handler()
        
        if user_id:
            langfuse_handler.user_id = user_id
        
        prompt, context, top_docs = self.__rag(query, augment_query, rerank, top_k)
        
        chain = prompt | self.ollama_model
        
        config = {
            "callbacks": [langfuse_handler],
            "metadata": {
            "user_id": langfuse_handler.user_id,
            "query_type": "rag",
            "augmented": augment_query,
            "reranked": rerank,
            "top_k": top_k
            }
        }
        
        response = chain.invoke({"question": query, "context": context}, config=config).content
        
        trace_id = langfuse_context.get_current_trace_id()
        
        return trace_id, top_docs, response


    @observe()
    def stream_call(self, query: str, user_id: str = None, 
                    augment_query: bool = True, rerank: bool = True, 
                    top_k: int = 3):
        langfuse_handler = langfuse_context.get_current_langchain_handler()
        
        if user_id:
            langfuse_handler.user_id = user_id
        
        prompt, context, top_docs = self.__rag(query, augment_query, rerank, top_k)
        
        chain = prompt | self.ollama_model
        
        config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "user_id": langfuse_handler.user_id,
                "query_type": "rag",
                "augmented": augment_query,
                "reranked": rerank,
                "top_k": top_k
            }
        }
        
        def stream_response():
            for chunk in chain.stream({"question": query, "context": context}, config=config):
                yield chunk.content

        trace_id = langfuse_context.get_current_trace_id()
        
        return trace_id, top_docs, stream_response()

    
    @observe()
    def ollama_inference(query: str, user_id: str = None, model: str = None, temperature: float = 0.8):
        langfuse_handler = langfuse_context.get_current_langchain_handler()
        
        if model:
            llm = ChatOllama(model=model, temperature=temperature, num_ctx=4096)
        
        if user_id:
            langfuse_handler.user_id = user_id
        
        config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "user_id": langfuse_handler.user_id,
                "query_type": "simple inference",
            }
        }
        
        def stream_response():
            for chunk in llm.stream(query, config=config):
                yield chunk.content

        trace_id = langfuse_context.get_current_trace_id()
        
        return trace_id, stream_response()
        
    