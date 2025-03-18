#!/bin/sh
# Installation script for mcp-server-qdrant using LangChain for embeddings

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

# Copy our LangChain embedding provider
echo "Installing LangChain embedding provider..."
mkdir -p /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cp /root/source/mcp/output/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/langchain_embed.py /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cp /root/source/mcp/output/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/minimal_embed.py /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/

# Update the embedding types to include LangChain provider
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/types.py << 'EOF'
from enum import Enum


class EmbeddingProviderType(Enum):
    MINIMAL = "minimal"
    LANGCHAIN = "langchain"
EOF

# Update the factory to use LangChain by default
cat > /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/factory.py << 'EOF'
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
EOF

# Update settings to use LangChain by default
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
        default=EmbeddingProviderType.LANGCHAIN,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
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

# Update pyproject.toml and setup.py to include langchain dependencies
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

[tool.uv]
dev-dependencies = [
    "pre-commit>=4.1.0",
    "pyright>=1.1.389",
    "pytest>=8.3.3",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.8.0"
]

[project.scripts]
mcp-server-qdrant = "mcp_server_qdrant.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
asyncio_mode = "auto"
EOF

cat > /tmp/mcp-server-qdrant/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="mcp-server-qdrant",
    version="0.1.0",
    description="MCP server for retrieving context from a Qdrant vector database",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "mcp[cli]>=1.3.0",
        "qdrant-client>=1.12.0",
        "pydantic>=2.10.6",
        "pydantic-settings>=2.0.0", 
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "langchain-community>=0.0.16",
    ],
    entry_points={
        "console_scripts": [
            "mcp-server-qdrant=mcp_server_qdrant.main:main",
        ],
    },
)
EOF

# Create a simple run script
cat > /tmp/mcp-server-qdrant/run.sh << 'EOF'
#!/bin/sh
# Script to run mcp-server-qdrant with LangChain embeddings

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