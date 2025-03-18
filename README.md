# MCP Server for Qdrant

A Machine Control Protocol (MCP) server for storing and retrieving information from a Qdrant vector database.

## Features

- Store text information with optional metadata in Qdrant
- Semantic search for stored information
- FastEmbed integration for text embeddings
- Environment-based configuration
- Docker support

## Installation

### Using pip

```bash
pip install mcp-server-qdrant
```

### From source

```bash
git clone https://github.com/your-org/mcp-server-qdrant.git
cd mcp-server-qdrant
make setup
```

## Configuration

Configuration is done through environment variables. You can create a `.env` file based on the `.env.example` file:

```bash
cp .env.example .env
```

Edit the `.env` file to configure the server:

```
# Qdrant configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-api-key

# Collection name
COLLECTION_NAME=memories

# Embedding provider configuration
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Usage

### Running locally

```bash
python -m mcp_server_qdrant.main
```

Or using the make command:

```bash
make run
```

### Docker

```bash
docker-compose up
```

## Tools

The MCP server provides the following tools:

### qdrant-store

Stores information in the Qdrant database.

```
information: The text to store
metadata: Optional JSON metadata to associate with the text
```

### qdrant-find

Searches for information in the Qdrant database using semantic search.

```
query: The search query
```

## Development

### Testing

```bash
make test
```

### Formatting

```bash
make format
```

### Linting

```bash
make lint
```

### Building

```bash
make build
```

## License

Apache License 2.0 