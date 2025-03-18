.PHONY: setup test format lint run build docker-build docker-run

setup:
	pip install uv
	uv pip install -e .
	uv pip install 'pytest>=8.0.0' 'pre-commit>=4.0.0' 'ruff>=0.8.0'

test:
	pytest

format:
	ruff format src tests

lint:
	ruff check src tests

run:
	python -m mcp_server_qdrant.main

build:
	uv pip install build
	python -m build

docker-build:
	docker build -t mcp-server-qdrant .

docker-run:
	docker-compose up 