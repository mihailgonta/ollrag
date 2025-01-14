import chromadb
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from sentence_transformers import CrossEncoder
from .utils import openai_token_count

class OllamaRag:
    def __init__(self, embeddings_model: str, ollama_model: str, temperature: float, collection_name: str, n_chunks: int = 5):
        self.temperature = temperature
        self.collection_name = collection_name
        self.n_chunks = n_chunks
        self.embeddings_model = OllamaEmbeddings(model=embeddings_model)
        self.ollama_model = OllamaLLM(model=ollama_model, temperature=temperature, verbose=True)
        self.chroma_client = chromadb.PersistentClient(path="data/chroma")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    
    def __get_chat_prompt(self, context: str, query: str):
        SYSTEM_TEMPLATE = """
        You are an AI assistant that provides answers strictly based on the provided context by the user.
        If the question cannot be answered using the given context, respond with: 'I don't know.'
        If no context is provided, also respond with: 'I don't know.'
        Do not add information or make assumptions beyond the given context.
        """
        
        HUMAN_MESSAGE_TEMPLATE = """
        Answer the question based on the provided context below:
        {context}
        --
        Here is the question:
        {question}
        """
        
        human_message_template = HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE_TEMPLATE)
        
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_TEMPLATE),
            human_message_template
        ])
        
        prompt = chat_prompt.format(context=context, question=query)
        
        return prompt

    
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
        
        response = self.ollama_model.invoke(prompt)
        
        questions = response.split("\n")
        
        return questions
    
    
    def __query_chroma(self, embedded_queries):
        chroma_collection = self.chroma_client.get_collection(name=self.collection_name)
        results = chroma_collection.query(query_embeddings=embedded_queries, n_results=self.n_chunks, include=['documents'])
        
        retrieved_documents = results['documents']
        
        unique_documents = set()
        for documents in retrieved_documents:
            for document in documents:
                unique_documents.add(document)
        
        return unique_documents
    
    
    def __embed_query(self, query: str, augment_query: bool):
        queries = [query]
        
        if augment_query:
            # Generating more queries based on the initial query
            augmented_queries = self.__augment_multiple_query(query)
            queries.extend(augmented_queries)
        
        # Embedding all queries in a single list 
        embedded_query = [self.embeddings_model.embed_query(query) for query in queries]

        return embedded_query
    

    def __rerank(self, query: str, documents: list):
        pairs = [[query, doc] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        
        doc_scores = list(zip(documents, scores))
        
        sorted_doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        
        return sorted_doc_scores
    
    
    def __build_context(self, documents: list, top_k: int):
        l = len(documents)
        n = max(min(l, top_k), 0)
        
        top_docs = [doc for doc, score in documents[:n]]
        
        context = "\n\n - -\n\n".join(top_docs)
        
        return context
    
    
    def __rag(self, query: str, augment_query: bool, top_k: int = 3):
        embedded_query = self.__embed_query(query, augment_query)
        
        documents = self.__query_chroma(embedded_query)

        ranked_docs = self.__rerank(query, documents)
        
        context = self.__build_context(ranked_docs, top_k)
        
        prompt = self.__get_chat_prompt(context=context, query=query)
        
        return prompt

    
    def call(self, query, augment_query: bool = True, top_k: int = 3):
        prompt = self.__rag(query, augment_query, top_k)
        response = self.ollama_model.invoke(prompt)
        return response


    def stream_call(self, query, augment_query: bool = True, top_k: int = 3):
        prompt = self.__rag(query, augment_query, top_k)
        for chunk in self.ollama_model.stream(prompt):
            yield chunk
    
    