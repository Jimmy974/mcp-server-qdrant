import asyncio
from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np

from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class SentenceTransformersProvider(EmbeddingProvider):
    """
    SentenceTransformers implementation of the embedding provider.
    :param model_name: The name of the SentenceTransformer model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since SentenceTransformer is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(documents)
        )
        # Convert numpy arrays to lists
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since SentenceTransformer is synchronous
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(query)
        )
        # Convert numpy array to list
        return embedding.tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        """
        model_name = self.model_name.split("/")[-1].lower()
        return f"st-{model_name}" 