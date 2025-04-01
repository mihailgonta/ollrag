RAG = """
You are an agent who provide answers strictly based on the provided context below:
{context}
--
Answer the following question based on the above context:{query}

Your responses should be concise, straight to the point, avoid filling words.
If the question cannot be answered using the given context, respond with: 'I don't know.'
If no context is provided, also respond with: 'I don't know.'
Do not add information or make assumptions beyond the given context.
"""

LLM_RERANK = """
You are a document reranking system. Your task is to evaluate how relevant each document is to the given query.
Assign a relevance score from 0-10 for each document, where 10 means highly relevant and 0 means not relevant at all.

query: {query}

document: {document}

Relevance score (0-10):
"""


DIRECT = """
You are a helpful AI assistant. Please answer the following question:
{query}
"""


QUERY_AUGMENTATION = """
You are a general purpose research assistant.
Do not number the questions.
Your users are asking questions on different topics.
Suggest up to five additional related questions to help them find the information they need, for the provided question.
Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
Make sure they are complete questions, and that they are related to the original question.
Output one question per line. Do not number the questions.

User question: {query}
"""
