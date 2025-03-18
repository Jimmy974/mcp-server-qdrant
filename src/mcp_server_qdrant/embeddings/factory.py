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
    # Try to use LangChain provider by default
    try:
        # Import here to avoid circular imports
        from mcp_server_qdrant.embeddings.langchain_embed import LangChainEmbedProvider
        logger.info(f"Creating LangChain embedding provider")
        return LangChainEmbedProvider(settings.model_name)
    except ImportError as e:
        logger.error(f"Failed to import LangChain provider: {e}")
        logger.info("Falling back to minimal provider")
        
        # Fall back to minimal provider
        try:
            from mcp_server_qdrant.embeddings.minimal_embed import MinimalEmbedProvider
            logger.info(f"Creating minimal embedding provider")
            return MinimalEmbedProvider('minimal')
        except ImportError as e:
            logger.error(f"Failed to import minimal provider: {e}")
            raise ValueError(
                "No embedding providers are available. "
                "This is a critical error as at least one provider is required."
            ) 