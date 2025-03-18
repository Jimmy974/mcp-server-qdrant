"""
Minimal embedding provider with no external ML dependencies.
Uses simple text processing techniques to create embeddings.
"""
import asyncio
import logging
import hashlib
import numpy as np
import re
import math
from typing import List, Dict, Any, Tuple
from collections import Counter

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

# Configure logging
logger = logging.getLogger(__name__)

class MinimalEmbedProvider(EmbeddingProvider):
    """
    Minimal embedding provider that works with no ML dependencies.
    Uses a combination of character n-grams and word frequencies for embedding.
    
    :param model_name: Ignored, but required by the interface (can be any string).
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vector_size = 384  # Fixed size for our embeddings
        self.vocab_size = 10000  # Maximum vocabulary size to consider
        self.ngram_ranges = [(1, 1), (2, 2), (3, 3)]  # Unigrams, bigrams, trigrams
        
        # Precomputed prime numbers for hashing
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        logger.info(f"Initialized MinimalEmbedProvider with vector size {self.vector_size}")

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        # Strip whitespace
        return text.strip()
    
    def _extract_features(self, text: str) -> Tuple[List[str], Counter]:
        """Extract features from text (ngrams and word frequencies)."""
        text = self._preprocess_text(text)
        tokens = text.split()
        
        # Word frequency features
        word_freq = Counter(tokens)
        
        # Character n-gram features
        ngrams = []
        for n_min, n_max in self.ngram_ranges:
            for n in range(n_min, n_max + 1):
                for i in range(len(text) - n + 1):
                    ngrams.append(text[i:i+n])
        
        return ngrams, word_freq
    
    def _compute_hash_embedding(self, features: List[str], word_freq: Counter) -> np.ndarray:
        """Compute embedding using a locality-sensitive hashing approach."""
        embedding = np.zeros(self.vector_size, dtype=np.float32)
        
        # Process up to vocab_size features to limit computation
        for i, feature in enumerate(features[:self.vocab_size]):
            # Use multiple hash functions for better distribution
            for j, prime in enumerate(self.primes):
                # Simple hash function using different primes
                h = int(hashlib.md5(feature.encode()).hexdigest(), 16) % prime
                idx = (h + j * prime) % self.vector_size
                embedding[idx] += 1.0
        
        # Add weighted word frequency components
        for word, count in word_freq.most_common(1000):
            # Get a hash for the word
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            # Use a different part of the vector for word frequencies
            idx = h % (self.vector_size // 4)
            # Add weighted by log frequency
            embedding[idx] += math.log(count + 1)
            
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    async def _embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        if not text or not text.strip():
            return [0.0] * self.vector_size
        
        # Extract features and compute embedding
        features, word_freq = self._extract_features(text)
        embedding = self._compute_hash_embedding(features, word_freq)
        
        return embedding.tolist()

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        if not documents:
            return []
        
        embeddings = []
        for doc in documents:
            embedding = await self._embed_text(doc)
            embeddings.append(embedding)
            
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        return await self._embed_text(query)

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        """
        return "minimal-embed" 