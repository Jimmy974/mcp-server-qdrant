FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy requirements files
COPY pyproject.toml ./
COPY src ./src

# Install dependencies
RUN uv pip install -e . && \
    if [ -f uv.lock ]; then uv pip install --no-deps -r uv.lock; fi

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the MCP server
CMD ["python", "-m", "mcp_server_qdrant.main"] 