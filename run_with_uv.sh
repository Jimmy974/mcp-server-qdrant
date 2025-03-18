#!/bin/sh
# Script to run mcp-server-qdrant using uv in Alpine

# Make sure we're in the right directory
cd /tmp/mcp-server-qdrant

# Set environment variables
export EMBEDDING_PROVIDER=alpine
export EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export COLLECTION_NAME=memories
export LOG_LEVEL=INFO

# Run the server using uv
echo "Starting mcp-server-qdrant with uv..."
uv --directory /tmp/mcp-server-qdrant/src/mcp_server_qdrant run mcp-server-qdrant 