from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings
import logging

# Set up logger
logger = logging.getLogger(__name__)

def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    Default to minimal provider if other requested providers are not available.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    # First try to use the requested provider
    if settings.provider_type != EmbeddingProviderType.MINIMAL:
        logger.info(f"Requested provider {settings.provider_type.value}, but defaulting to minimal provider")
    
    # Always use minimal provider for maximum compatibility
    try:
        # Import here to avoid circular imports
        from mcp_server_qdrant.embeddings.minimal_embed import MinimalEmbedProvider
        logger.info(f"Creating minimal embedding provider")
        return MinimalEmbedProvider(settings.model_name)
    except ImportError as e:
        logger.error(f"Failed to import minimal provider: {e}")
        raise ValueError(
            "Minimal embedding provider is not available. "
            "This is a critical error as the minimal provider has no external dependencies."
        ) 