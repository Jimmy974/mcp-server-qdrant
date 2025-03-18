"""
LangChain-based embedding provider.
Uses LangChain's embedding models for maximum compatibility.
"""
import asyncio
import logging
from typing import List, Optional

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# Configure logging
logger = logging.getLogger(__name__)

class LangChainEmbedProvider(EmbeddingProvider):
    """
    LangChain-based embedding provider.
    Uses LangChain's embedding utilities for compatibility across platforms.
    
    :param model_name: The name of the model to use (if supported by LangChain).
    """

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.embedding_model = None
        self.vector_size = 384  # Default size
        
        try:
            # Import LangChain components
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            from langchain_community.embeddings import DeterministicFakeEmbedding
            
            # Try to use a lightweight HuggingFace model
            try:
                self.embedding_model = HuggingFaceBgeEmbeddings(
                    model_name="BAAI/bge-small-en-v1.5",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Successfully initialized HuggingFace BGE embedding model")
            except Exception as e:
                logger.warning(f"Failed to initialize BGE model: {e}")
                # Fall back to fake embeddings for testing/compatibility
                logger.info("Falling back to deterministic embeddings")
                self.embedding_model = DeterministicFakeEmbedding(size=384)
                
        except ImportError as e:
            logger.warning(f"LangChain not available: {e}. Using deterministic embeddings.")
            # Use our own minimal implementation if LangChain isn't available
            self._init_minimal_embedding()
    
    def _init_minimal_embedding(self):
        """Initialize minimal embedding if LangChain is not available."""
        import hashlib
        import numpy as np
        
        # Create a simple deterministic embedding function
        def fake_embed(text: str) -> List[float]:
            # Create a deterministic vector based on text hash
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            vec = np.random.normal(0, 1, self.vector_size)
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        
        # Create a wrapper to mimic LangChain interface
        class MinimalEmbedding:
            def embed_documents(self, documents: List[str]) -> List[List[float]]:
                return [fake_embed(doc) for doc in documents]
                
            def embed_query(self, query: str) -> List[float]:
                return fake_embed(query)
        
        self.embedding_model = MinimalEmbedding()
        logger.info("Using minimal deterministic embedding as fallback")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        if not documents:
            return []
        
        # Run in a thread pool since LangChain is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.embedding_model.embed_documents, documents
        )
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        if not query or not query.strip():
            # Return a zero vector of appropriate dimension to avoid errors
            return [0.0] * self.vector_size
        
        # Run in a thread pool since LangChain is synchronous
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.embedding_model.embed_query, query
        )
        return embedding

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        """
        return "langchain-embed" 