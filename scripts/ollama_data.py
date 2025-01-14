import os
import uuid
import chromadb
import configparser
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader

from .chunking.cluster_semantic_chunker import ClusterSemanticChunker

class OllamaDb:
    def __init__(self, embedding_model: str):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.config = configparser.ConfigParser()
        self.chroma_client = self.__get_chroma_client()
    
    
    def __get_chroma_client(self):
        try:
            self.config.read('config.ini')
            chroma_path = self.config['chroma']['path']
            print(f"ChromaDB path: {chroma_path}")
            
            return chromadb.PersistentClient(chroma_path)
        except Exception as e:
            raise ValueError(f"Couldn't connect to ChromaDB. Error: {str(e)}") from e
            

    def get_chroma_collection(self, name: str):
        try:
            collection = self.chroma_client.get_collection(name)
            return collection
        except Exception as e:
            raise ValueError(f"Collection '{name}' does not exist. Error: {str(e)}") from e
    

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        docs = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                # PyMuPDFLoader creates a document per page
                # Concatenate them to preserve context across pages
                loader = PyMuPDFLoader(file_path)
                multiple_documents = loader.load()
                page_content = " ".join(doc.page_content for doc in multiple_documents)
                document = Document(page_content=str(page_content), metadata={"source": file_path})
            elif file_path.endswith('.txt') or file_path.endswith('.md'):
                document = TextLoader(file_path).load()
            else:
                print(f"Skipping unsupported file type: {file_path}")
                continue
            docs.append(document)
            
        return docs
            

    def chunk_documents(self, documents: List[Document]):
        text_splitter = ClusterSemanticChunker(self.embeddings.embed_documents)
        chunks = text_splitter.split_documents(documents)
        
        return chunks


    def create_collection(self, chunks: List[Document], name: str):
        try:
            collection = self.chroma_client.get_or_create_collection(name)
            
            chunk_ids = [f"{uuid.uuid4()}" for i in range(len(chunks))]
            chunk_embeddings = self.embeddings.embed_documents([chunk.page_content for chunk in chunks])
            chunks_metadata = [chunk.metadata for chunk in chunks if chunk.metadata]
            chunks_content = [chunk.page_content for chunk in chunks]

            collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                metadatas=chunks_metadata,
                documents=chunks_content,
            )
        except Exception as e:
            raise ValueError(f"Couldn't create collection '{name}'. Error: {str(e)}.") from e
            