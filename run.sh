#!/bin/bash
set -e

# Create a virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e .

# Check if .env file exists, if not create from example
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env file if needed."
fi

# Run the MCP server
echo "Starting MCP server..."
python -m mcp_server_qdrant.main 