#!/bin/sh
# Alpine Linux installation script for mcp-server-qdrant

# Ensure Python and other required packages are installed
apk add --update python3 py3-pip git build-base python3-dev openblas-dev

# Create a virtual environment 
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install onnxruntime and fastembed for embeddings
pip install "onnxruntime>=1.14.0" fastembed numpy

# Install tokenizers for better text processing
pip install tokenizers

# Install the minimum dependencies for mcp-server-qdrant
pip install --no-deps mcp[cli]>=1.3.0 qdrant-client>=1.12.0 pydantic>=2.10.6 pydantic-settings>=2.0.0 python-dotenv>=1.0.0

# Install the package itself in development mode
pip install -e .

# Create a .env file that uses the Alpine-compatible embedding provider
cat > .env << EOF
EMBEDDING_PROVIDER=alpine
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
COLLECTION_NAME=memories
LOG_LEVEL=INFO
EOF

echo "Installation complete. Use the following command to start the server:"
echo "source /opt/venv/bin/activate && python -m mcp_server_qdrant.main" 