from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
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
        from mcp_server_qdrant.embeddings.sentence_transformers import SentenceTransformersProvider
        return SentenceTransformersProvider(settings.model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}") 