#!/bin/sh
# Activate the virtual environment
source /opt/venv/bin/activate

# Set any additional environment variables if needed
export EMBEDDING_PROVIDER=fastembed
export EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export COLLECTION_NAME=memories
export LOG_LEVEL=INFO

# Start the MCP server
cd /tmp/mcp-server-qdrant
python -m mcp_server_qdrant.main 