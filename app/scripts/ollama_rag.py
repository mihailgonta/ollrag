import os
from dataclasses import dataclass
from typing import Optional, Generator, Tuple, List, Union

from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from langfuse.decorators import observe, langfuse_context

from scripts import prompt_templates


@dataclass
class RAGConfig:
    augment_query: bool = False
    rerank: bool = False
    top_k: int = 3
    n_chunks: int = 10


@dataclass
class QueryResult:
    trace_id: str
    documents: Optional[List[Document]]
    content: Union[str, Generator]


class ScoreResponse(BaseModel):
    """Always use this tool to structure your response to the user."""

    score: int = Field(description="Relevance score from 0-10 of the document.")


class OllamaAssistant:
    def __init__(
        self,
        chat_model: BaseChatModel,
        embeddings: Optional[Embeddings] = None,
        collection_name: Optional[str] = None,
        rag_config: Optional[RAGConfig] = None,
    ):
        self.chat_model = chat_model
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.rag_config = rag_config or RAGConfig()

        if embeddings and collection_name:
            self.chroma_path = os.path.join("..", "data", "chroma")
            self.collection = Chroma(
                collection_name=collection_name, persist_directory=self.chroma_path
            )

    def __get_chat_prompt(self, use_rag: bool):
        template = prompt_templates.RAG if use_rag else prompt_templates.DIRECT
        return ChatPromptTemplate.from_template(template)

    def __augment_query(self, query):
        prompt_template = ChatPromptTemplate.from_template(
            prompt_templates.QUERY_AUGMENTATION
        )
        prompt = prompt_template.format(question=query)
        response = self.chat_model.invoke(prompt).content

        return [q.strip() for q in response.split("\n") if q.strip()]

    def __embed_and_query(self, query: str) -> List[Document]:
        if not self.embeddings or not self.collection:
            raise ValueError(
                "RAG functionality requires embeddings and collection_name"
            )

        queries = (
            self.__augment_query(query) if self.rag_config.augment_query else [query]
        )
        embedded_queries = [self.embeddings.embed_query(q) for q in queries]

        results = []
        for embedding in embedded_queries:
            results.extend(
                self.collection.similarity_search_by_vector(
                    embedding=embedding, k=self.rag_config.n_chunks
                )
            )

        return list({doc.page_content: doc for doc in results}.values())

    def __llm_rerank(self, query: str, documents: list[Document]):
        rerank_prompt = ChatPromptTemplate.from_template(prompt_templates.LLM_RERANK)

        chain = rerank_prompt | self.chat_model.with_structured_output(ScoreResponse)

        for doc in documents:
            doc.metadata["score"] = chain.invoke(
                {"query": query, "document": doc.page_content}
            )

        sorted_documents = sorted(
            documents, key=lambda x: x.metadata["score"].score, reverse=True
        )

        return sorted_documents[: self.rag_config.top_k]

    def __prepare_response_config(
        self, user_id: Optional[str], query_type: str
    ) -> dict:
        langfuse_handler = langfuse_context.get_current_langchain_handler()

        if user_id:
            langfuse_handler.user_id = user_id

        metadata = {
            "user_id": langfuse_handler.user_id,
            "query_type": query_type,
        }

        if query_type == "rag":
            metadata.update({"rag_config": self.rag_config})

        return {"callbacks": [langfuse_handler], "metadata": metadata}

    def __prepare_chain_input(
        self, query: str, use_rag: bool
    ) -> Tuple[dict, Optional[List[Document]]]:
        if use_rag:
            if not (self.embeddings and self.collection_name):
                raise ValueError(
                    "RAG functionality requires embeddings and collection_name"
                )

            documents = self.__embed_and_query(query)

            if self.rag_config.rerank:
                documents = self.__llm_rerank(query, documents)

            context = "\n\n---\n\n".join(doc.page_content for doc in documents)

            return {"question": query, "context": context}, documents

        return {"question": query}, None

    def __process_query(
        self, query: str, user_id: Optional[str], use_rag: bool, chain_method: str
    ) -> QueryResult:
        config = self.__prepare_response_config(user_id, "rag" if use_rag else "direct")

        documents = []

        chain_input, documents = self.__prepare_chain_input(query, use_rag)

        prompt = self.__get_chat_prompt(use_rag)
        chain = prompt | self.chat_model

        method = getattr(chain, chain_method)

        if chain_method == "stream":
            content = (chunk.content for chunk in method(chain_input, config=config))
        else:
            content = method(chain_input, config=config).content

        return QueryResult(
            trace_id=langfuse_context.get_current_trace_id(),
            documents=documents,
            content=content,
        )

    @observe()
    def query(
        self, query: str, user_id: Optional[str] = None, use_rag: bool = False
    ) -> QueryResult:
        return self.__process_query(query, user_id, use_rag, "invoke")

    @observe()
    def stream_query(
        self, query: str, user_id: Optional[str] = None, use_rag: bool = False
    ) -> QueryResult:
        return self.__process_query(query, user_id, use_rag, "stream")
