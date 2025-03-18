import pytest
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


@pytest.mark.asyncio
async def test_fastembed_provider():
    """Test the FastEmbed provider."""
    # Create a settings object with the FastEmbed provider
    settings = EmbeddingProviderSettings(
        provider_type=EmbeddingProviderType.FASTEMBED,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    
    # Create the embedding provider
    provider = create_embedding_provider(settings)
    
    # Test embedding a query
    query = "This is a test query"
    embedding = await provider.embed_query(query)
    
    # Check that the embedding is a list of floats
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    
    # Test embedding documents
    documents = ["This is document 1", "This is document 2"]
    embeddings = await provider.embed_documents(documents)
    
    # Check that the embeddings are a list of lists of floats
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(documents)
    assert all(isinstance(x, list) for x in embeddings)
    assert all(all(isinstance(y, float) for y in x) for x in embeddings)
    
    # Check that the vector name is as expected
    vector_name = provider.get_vector_name()
    assert vector_name.startswith("fast-") 