#!/bin/sh
# Script to run mcp-server-qdrant with minimal embedding provider

# Set environment variables
export COLLECTION_NAME=memories
export LOG_LEVEL=INFO

# Install only numpy as dependency
pip install numpy

# Run the server
cd /tmp/mcp-server-qdrant
python -m mcp_server_qdrant.main 