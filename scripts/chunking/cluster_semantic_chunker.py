import numpy as np

from .base_chunker import BaseChunker
from typing import List, Optional, Iterable

from scripts.chunking import RecursiveTokenChunker

from langchain_core.documents import Document
from scripts.utils import get_openai_embedding_function, openai_token_count

from langchain_community.utils.math import cosine_similarity

class ClusterSemanticChunker(BaseChunker):
    def __init__(self, embedding_function=None, max_chunk_size=400, min_chunk_size=50, length_function=openai_token_count):
        self.splitter = RecursiveTokenChunker(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
            length_function=openai_token_count,
            separators = ["\n\n", "\n", ".", "?", "!", " ", ""]
            )
        
        if embedding_function is None:
            embedding_function = get_openai_embedding_function()
        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size//min_chunk_size
        self.embedding_function = embedding_function
        
    def _get_similarity_matrix(self, embedding_function, sentences):
        BATCH_SIZE = 500
        N = len(sentences)
        embedding_matrix = []

        for i in range(0, N, BATCH_SIZE):
            batch_sentences = sentences[i:i + BATCH_SIZE]
            embeddings = embedding_function(batch_sentences)

            # Check if embeddings are valid
            if embeddings is None or len(embeddings) == 0:
                raise ValueError(f"Embedding function returned no embeddings for batch {i // BATCH_SIZE}.")

            # Append the batch embeddings to the embedding matrix
            embedding_matrix.extend(embeddings)

        if len(embedding_matrix) == 0:
            raise ValueError("Failed to generate embeddings for the given sentences.")

        # Convert embedding_matrix to numpy array
        embedding_matrix = np.array(embedding_matrix)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(embedding_matrix, embedding_matrix)

        return similarity_matrix

    def _calculate_reward(self, matrix, start, end):
        sub_matrix = matrix[start:end+1, start:end+1]
        return np.sum(sub_matrix)

    def _optimal_segmentation(self, matrix, max_cluster_size, window_size=3):
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 1 to avoid trivial solutions

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    # local_density = calculate_local_density(matrix, i, window_size)
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    # Adjust reward based on local density
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters
        
    def split_text(self, text: str) -> List[str]:
        sentences = self.splitter.split_text(text)

        similarity_matrix = self._get_similarity_matrix(self.embedding_function, sentences)

        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        docs = [' '.join(sentences[start:end+1]) for start, end in clusters]
        
        return docs
    
    def split_documents(self, documents: Iterable[Document], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        
        docs = []
        
        for doc in documents:
            if doc.page_content:
                chunks = self.split_text(doc.page_content)
                docs.extend([Document(page_content=chunk, metadata={'source': doc.metadata['source']}) for chunk in chunks]) 
            
        return docs
 