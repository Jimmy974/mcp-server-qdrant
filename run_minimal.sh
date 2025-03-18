#!/bin/sh
# Script to run mcp-server-qdrant with minimal embedding provider

# Set environment variables
export COLLECTION_NAME=memories
export LOG_LEVEL=INFO

# Install only numpy as dependency
pip install numpy

# Clone the repo if not already present
if [ ! -d "/tmp/mcp-server-qdrant" ]; then
  git clone https://github.com/Jimmy974/mcp-server-qdrant.git /tmp/mcp-server-qdrant
  cd /tmp/mcp-server-qdrant
  git checkout ef795ae51801ac7bc875f0e1f9c3c3422c61d70b
fi

# Copy our custom files to the cloned repo
mkdir -p /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cp /root/source/mcp/output/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/minimal_embed.py /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cp /root/source/mcp/output/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/types.py /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cp /root/source/mcp/output/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/factory.py /tmp/mcp-server-qdrant/src/mcp_server_qdrant/embeddings/
cp /root/source/mcp/output/mcp-server-qdrant/src/mcp_server_qdrant/settings.py /tmp/mcp-server-qdrant/src/mcp_server_qdrant/
cp /root/source/mcp/output/mcp-server-qdrant/setup.py /tmp/mcp-server-qdrant/

# Install with pip (avoiding torch)
cd /tmp/mcp-server-qdrant
pip install --no-deps .
pip install qdrant-client>=1.12.0 pydantic>=2.10.6 pydantic-settings>=2.0.0 python-dotenv>=1.0.0 numpy>=1.24.0 mcp[cli]>=1.3.0

# Run the server
python -m mcp_server_qdrant.main 