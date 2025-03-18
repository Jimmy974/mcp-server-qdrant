"""
Direct LangChain embedding implementation without factory pattern.
"""
import asyncio
import logging
import hashlib
import numpy as np
from typing import List

# Configure logging
logger = logging.getLogger(__name__)

# Default vector size
VECTOR_SIZE = 384

class EmbeddingService:
    """
    Simple embedding service that uses LangChain directly without factory pattern.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create a singleton instance"""
        if cls._instance is None:
            cls._instance = EmbeddingService()
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding model."""
        self.embedding_model = None
        self.vector_size = VECTOR_SIZE
        
        # Try LangChain first
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
                self.embedding_model = DeterministicFakeEmbedding(size=VECTOR_SIZE)
                
        except ImportError as e:
            logger.warning(f"LangChain not available: {e}. Using hash-based embeddings.")
            # Use our own minimal implementation if LangChain isn't available
            self._init_minimal_embedding()
    
    def _init_minimal_embedding(self):
        """Initialize minimal hash-based embedding."""
        # Create a wrapper to mimic LangChain interface
        class HashEmbedding:
            def embed_documents(self, documents: List[str]) -> List[List[float]]:
                return [self._hash_embed(doc) for doc in documents]
                
            def embed_query(self, query: str) -> List[float]:
                return self._hash_embed(query)
                
            def _hash_embed(self, text: str) -> List[float]:
                # Create a deterministic vector based on text hash
                hash_obj = hashlib.md5(text.encode())
                seed = int(hash_obj.hexdigest(), 16) % (2**32)
                np.random.seed(seed)
                vec = np.random.normal(0, 1, VECTOR_SIZE)
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()
        
        self.embedding_model = HashEmbedding()
        logger.info("Using hash-based embedding as fallback")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        if not documents:
            return []
        
        # Run in a thread pool since embedding is synchronous
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
        
        # Run in a thread pool since embedding is synchronous
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.embedding_model.embed_query, query
        )
        return embedding

    def get_vector_name(self) -> str:
        """Return the name of the vector for the Qdrant collection."""
        return "embeddings"


# Create a singleton instance
embeddings = EmbeddingService.get_instance() 