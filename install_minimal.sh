#!/bin/sh
# Minimal installation script for mcp-server-qdrant on Alpine Linux

set -e  # Exit on any error

# Install core system packages
apk add --update python3 py3-pip git

# Install uv if needed
pip install uv

# Clone the repo if not already present
if [ ! -d "/tmp/mcp-server-qdrant" ]; then
  echo "Cloning repository..."
  git clone https://github.com/Jimmy974/mcp-server-qdrant.git /tmp/mcp-server-qdrant
  cd /tmp/mcp-server-qdrant
  git checkout ef795ae51801ac7bc875f0e1f9c3c3422c61d70b
else
  cd /tmp/mcp-server-qdrant
fi

# Install only the minimal dependencies with uv
echo "Installing minimal dependencies..."
uv pip install --system numpy
uv pip install --system --no-deps mcp[cli]>=1.3.0 qdrant-client>=1.12.0 pydantic>=2.10.6 pydantic-settings>=2.0.0 python-dotenv>=1.0.0

# Copy our custom minimal embedding provider
echo "Installing minimal embedding provider..."
mkdir -p /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/minimal_embed.py << 'EOF'
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
        self.vector_size = 512  # Fixed size for our embeddings
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
EOF

# Update the embedding types to include MINIMAL provider
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/types.py << 'EOF'
from enum import Enum


class EmbeddingProviderType(Enum):
    MINIMAL = "minimal"
EOF

# Update the factory to use our minimal provider
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/factory.py << 'EOF'
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings
import logging

# Set up logger
logger = logging.getLogger(__name__)

def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create a minimal embedding provider regardless of settings.
    :param settings: The settings for the embedding provider.
    :return: An instance of the minimal embedding provider.
    """
    # Always use minimal provider
    try:
        # Import here to avoid circular imports
        from mcp_server_qdrant.embeddings.minimal_embed import MinimalEmbedProvider
        logger.info(f"Creating minimal embedding provider")
        return MinimalEmbedProvider('minimal')
    except ImportError as e:
        logger.error(f"Failed to import minimal provider: {e}")
        raise ValueError(
            "Minimal embedding provider is not available. "
            "This is a critical error as the minimal provider has no external dependencies."
        )
EOF

# Update settings to use minimal provider by default
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/settings.py << 'EOF'
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user"
)


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.MINIMAL,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="minimal",
        validation_alias="EMBEDDING_MODEL",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: Optional[str] = Field(default=None, validation_alias="QDRANT_URL")
    api_key: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str = Field(default="memories", validation_alias="COLLECTION_NAME")
    local_path: Optional[str] = Field(
        default=None, validation_alias="QDRANT_LOCAL_PATH"
    )

    def get_qdrant_location(self) -> str:
        """
        Get the Qdrant location, either the URL or the local path.
        """
        return self.location or self.local_path
EOF

# Run the server with UV
echo "========================================================"
echo "Installation complete! Run with:"
echo "cd /tmp/mcp-server-qdrant && uv run -p /tmp/mcp-server-qdrant python -m mcp_server_qdrant.main"
echo "========================================================" 