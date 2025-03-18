import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_server_qdrant.qdrant import Entry, QdrantConnector
from mcp_server_qdrant.embeddings.base import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self):
        self.embed_documents_mock = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        self.embed_query_mock = AsyncMock(return_value=[0.1, 0.2, 0.3])

    async def embed_documents(self, documents):
        return await self.embed_documents_mock(documents)

    async def embed_query(self, query):
        return await self.embed_query_mock(query)

    def get_vector_name(self):
        return "mock-vector"


@pytest.fixture
def mock_qdrant_client():
    """Mock the AsyncQdrantClient."""
    with patch("mcp_server_qdrant.qdrant.AsyncQdrantClient") as mock:
        client = MagicMock()
        client.collection_exists = AsyncMock(return_value=True)
        client.search = AsyncMock(return_value=[
            MagicMock(payload={"document": "test content", "metadata": {"key": "value"}})
        ])
        client.upsert = AsyncMock()
        mock.return_value = client
        yield client


@pytest.fixture
def qdrant_connector(mock_qdrant_client):
    """Create a QdrantConnector with mocked dependencies."""
    embedding_provider = MockEmbeddingProvider()
    return QdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_name="test-collection",
        embedding_provider=embedding_provider,
    )


@pytest.mark.asyncio
async def test_store(qdrant_connector, mock_qdrant_client):
    """Test storing an entry."""
    entry = Entry(content="Test information", metadata={"key": "value"})
    await qdrant_connector.store(entry)

    # Verify the client was called correctly
    mock_qdrant_client.upsert.assert_called_once()
    call_args = mock_qdrant_client.upsert.call_args[1]
    assert call_args["collection_name"] == "test-collection"
    assert len(call_args["points"]) == 1
    assert call_args["points"][0].payload["document"] == "Test information"
    assert call_args["points"][0].payload["metadata"] == {"key": "value"}


@pytest.mark.asyncio
async def test_search(qdrant_connector, mock_qdrant_client):
    """Test searching for entries."""
    results = await qdrant_connector.search("test query")

    # Verify the client was called correctly
    mock_qdrant_client.search.assert_called_once()
    call_args = mock_qdrant_client.search.call_args[1]
    assert call_args["collection_name"] == "test-collection"
    assert call_args["query_vector"].name == "mock-vector"
    assert call_args["query_vector"].vector == [0.1, 0.2, 0.3]

    # Check the results
    assert len(results) == 1
    assert results[0].content == "test content"
    assert results[0].metadata == {"key": "value"} 