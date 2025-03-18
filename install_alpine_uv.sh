#!/bin/sh
# Alpine Linux installation script for mcp-server-qdrant using uv

# Ensure Python and other required packages are installed
apk add --update python3 py3-pip git build-base python3-dev openblas-dev

# Install uv if not already installed
pip install uv

# Install dependencies using uv in the target directory
cd /tmp/mcp-server-qdrant

# Install core dependencies with uv
uv pip install --system "onnxruntime>=1.14.0" fastembed numpy tokenizers
uv pip install --system mcp[cli]>=1.3.0 qdrant-client>=1.12.0 pydantic>=2.10.6 pydantic-settings>=2.0.0 python-dotenv>=1.0.0

# Copy the Alpine embedding provider files
mkdir -p /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cp /root/source/mcp/output/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/alpine_compat.py /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/

# Update the embedding types to include Alpine provider
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/types.py << EOF
from enum import Enum


class EmbeddingProviderType(Enum):
    FASTEMBED = "fastembed"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    ALPINE = "alpine"
EOF

# Update the factory to use our Alpine provider
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/factory.py << EOF
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings
import logging

# Set up logger
logger = logging.getLogger(__name__)

def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.ALPINE:
        try:
            from mcp_server_qdrant.embeddings.alpine_compat import AlpineEmbedProvider
            logger.info(f"Creating Alpine-compatible embedding provider with model {settings.model_name}")
            return AlpineEmbedProvider(settings.model_name)
        except ImportError as e:
            logger.error(f"Failed to import Alpine provider: {e}")
            logger.info("Falling back to FastEmbed provider")
            # Fall back to FastEmbed if Alpine provider fails
            settings.provider_type = EmbeddingProviderType.FASTEMBED
    
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        try:
            from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
            return FastEmbedProvider(settings.model_name)
        except ImportError:
            raise ValueError(
                "FastEmbed provider is not available. "
                "Please install fastembed package or use another provider."
            )
    elif settings.provider_type == EmbeddingProviderType.SENTENCE_TRANSFORMERS:
        try:
            from mcp_server_qdrant.embeddings.sentence_transformers import SentenceTransformersProvider
            return SentenceTransformersProvider(settings.model_name)
        except ImportError:
            raise ValueError(
                "SentenceTransformers provider is not available. "
                "Please install sentence-transformers package or use another provider."
            )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
EOF

# Create a .env file that uses the Alpine-compatible provider
cat > /tmp/mcp-server-qdrant/.env << EOF
EMBEDDING_PROVIDER=alpine
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
COLLECTION_NAME=memories
LOG_LEVEL=INFO
EOF

echo "Installation complete. Use the following command to run the server:"
echo "uv --directory /tmp/mcp-server-qdrant/src/mcp_server_qdrant run mcp-server-qdrant" 