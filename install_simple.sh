#!/bin/sh
# Simple installation script for mcp-server-qdrant using direct LangChain embeddings

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

# Install dependencies with uv
echo "Installing dependencies..."
uv pip install --system numpy
uv pip install --system langchain-community
uv pip install --system --no-deps mcp[cli]>=1.3.0 qdrant-client>=1.12.0 pydantic>=2.10.6 pydantic-settings>=2.0.0 python-dotenv>=1.0.0

# Create directory structure
echo "Setting up simplified embedding..."
mkdir -p /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/

# Create the direct LangChain embedding module
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/langchain_direct.py << 'EOF'
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
EOF

# Create the empty __init__.py file
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/__init__.py << 'EOF'
"""
Embedding utilities for the MCP Server Qdrant.
"""
EOF

# Update server.py to use direct LangChain embeddings
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/server.py << 'EOF'
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from mcp.server import Server
from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.langchain_direct import embeddings
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:  # noqa
    """
    Context manager to handle the lifespan of the server.
    This is used to configure the Qdrant connector.
    All the configuration is now loaded from the environment variables.
    Settings handle that for us.
    """
    try:
        # Use direct langchain embeddings
        logger.info("Using LangChain embeddings directly")

        qdrant_configuration = QdrantSettings()
        qdrant_connector = QdrantConnector(
            qdrant_configuration.location,
            qdrant_configuration.api_key,
            qdrant_configuration.collection_name,
            embeddings,
            qdrant_configuration.local_path,
        )
        logger.info(
            f"Connecting to Qdrant at {qdrant_configuration.get_qdrant_location()}"
        )

        yield {
            "embedding_service": embeddings,
            "qdrant_connector": qdrant_connector,
        }
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        pass


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
mcp = FastMCP("mcp-server-qdrant", lifespan=server_lifespan)

# Load the tool settings from the env variables, if they are set,
# or use the default values otherwise.
tool_settings = ToolSettings()


@mcp.tool(name="qdrant-store", description=tool_settings.tool_store_description)
async def store(
    ctx: Context,
    information: str,
    # The `metadata` parameter is defined as non-optional, but it can be None.
    # If we set it to be optional, some of the MCP clients, like Cursor, cannot
    # handle the optional parameter correctly.
    metadata: Metadata = None,
) -> str:
    """
    Store some information in Qdrant.
    :param ctx: The context for the request.
    :param information: The information to store.
    :param metadata: JSON metadata to store with the information, optional.
    :return: A message indicating that the information was stored.
    """
    await ctx.debug(f"Storing information {information} in Qdrant")
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    entry = Entry(content=information, metadata=metadata)
    await qdrant_connector.store(entry)
    return f"Remembered: {information}"


@mcp.tool(name="qdrant-find", description=tool_settings.tool_find_description)
async def find(ctx: Context, query: str) -> List[str]:
    """
    Find memories in Qdrant.
    :param ctx: The context for the request.
    :param query: The query to use for the search.
    :return: A list of entries found.
    """
    await ctx.debug(f"Finding results for query {query}")
    logger.info(f"Finding results for query {query}")
    qdrant_connector: QdrantConnector = ctx.request_context.lifespan_context[
        "qdrant_connector"
    ]
    entries = await qdrant_connector.search(query)
    if not entries:
        return [f"No information found for the query '{query}'"]
    content = [
        f"Results for the query '{query}'",
    ]
    for entry in entries:
        # Format the metadata as a JSON string and produce XML-like output
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        content.append(
            f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"
        )
    return content
EOF

# Update qdrant.py to work with the embedding service
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/qdrant.py << 'EOF'
import logging
import uuid
from typing import Any, Dict, Optional, List

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Optional[Metadata] = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use.
    :param embedding_service: The embedding service to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: str,
        embedding_service,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._collection_name = collection_name
        self._embedding_service = embedding_service
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )

    async def _ensure_collection_exists(self):
        """Ensure that the collection exists, creating it if necessary."""
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            # We'll get the vector size by embedding a sample text
            sample_vector = await self._embedding_service.embed_query("sample text")
            vector_size = len(sample_vector)

            # Use the vector name as defined in the embedding service
            vector_name = self._embedding_service.get_vector_name()
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

    async def store(self, entry: Entry):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        """
        await self._ensure_collection_exists()

        # Embed the document
        embeddings = await self._embedding_service.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_service.get_vector_name()
        payload = {"document": entry.content, "metadata": entry.metadata}
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def search(self, query: str) -> List[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :return: A list of entries found.
        """
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            return []

        # Embed the query
        query_vector = await self._embedding_service.embed_query(query)
        vector_name = self._embedding_service.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=models.NamedVector(name=vector_name, vector=query_vector),
            limit=10,
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results
        ]
EOF

# Update settings.py to remove embedding provider settings
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/settings.py << 'EOF'
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

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

# Update dependencies in pyproject.toml
cat > /tmp/mcp-server-qdrant/pyproject.toml << 'EOF'
[project]
name = "mcp-server-qdrant"
version = "0.1.0"
description = "MCP server for retrieving context from a Qdrant vector database"
readme = "README.md"
requires-python = ">=3.10"
license = "Apache-2.0"
dependencies = [
    "mcp[cli]>=1.3.0",
    "qdrant-client>=1.12.0",
    "pydantic>=2.10.6",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "numpy>=1.24.0",
    "langchain-community>=0.0.16",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project.scripts]
mcp-server-qdrant = "mcp_server_qdrant.main:main"
EOF

# Create a simple run script
cat > /tmp/mcp-server-qdrant/run.sh << 'EOF'
#!/bin/sh
# Script to run mcp-server-qdrant with direct LangChain embeddings

cd /tmp/mcp-server-qdrant
python -m mcp_server_qdrant.main
EOF

chmod +x /tmp/mcp-server-qdrant/run.sh

echo "========================================================"
echo "Installation complete! Run with:"
echo "cd /tmp/mcp-server-qdrant && uv run -p /tmp/mcp-server-qdrant python -m mcp_server_qdrant.main"
echo "or"
echo "/tmp/mcp-server-qdrant/run.sh"
echo "=========================================================" 