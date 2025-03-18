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
    ],
    entry_points={
        "console_scripts": [
            "mcp-server-qdrant=mcp_server_qdrant.main:main",
        ],
    },
) 