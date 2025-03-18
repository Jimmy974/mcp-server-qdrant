import logging
import uuid
from typing import Any, Dict, Optional, List

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Optional[Metadata] = None
    score: Optional[float] = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the collection to use.
    :param embedding_service: The embedding service to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: str,
        embedding_service,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._collection_name = collection_name
        self._embedding_service = embedding_service
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )

    async def _ensure_collection_exists(self):
        """Ensure that the collection exists, creating it if necessary."""
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            # We'll get the vector size by embedding a sample text
            sample_vector = await self._embedding_service.embed_query("sample text")
            vector_size = len(sample_vector)

            # Use the vector name as defined in the embedding service
            vector_name = self._embedding_service.get_vector_name()
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                },
            )

    async def store(self, entry: Entry):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        """
        await self._ensure_collection_exists()

        # Embed the document
        embeddings = await self._embedding_service.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_service.get_vector_name()
        payload = {"document": entry.content, "metadata": entry.metadata}
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )

    async def search(
        self, 
        query: str, 
        limit: int = 10, 
        score_threshold: float = 0.7,
        filter_metadata: Optional[Metadata] = None
    ) -> List[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param limit: Maximum number of results to return (default: 10).
        :param score_threshold: Minimum similarity score threshold (default: 0.7).
        :param filter_metadata: Optional metadata filter to narrow search results.
        :return: A list of entries found with their similarity scores.
        """
        collection_exists = await self._client.collection_exists(self._collection_name)
        if not collection_exists:
            return []

        # Embed the query
        query_vector = await self._embedding_service.embed_query(query)
        vector_name = self._embedding_service.get_vector_name()

        # Create filter if metadata filter is provided
        filter_condition = None
        if filter_metadata:
            filter_conditions = []
            for key, value in filter_metadata.items():
                if isinstance(value, list):
                    # Handle list values with 'any' condition
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    # Handle scalar values with 'match' condition
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )
            
            if filter_conditions:
                filter_condition = models.Filter(
                    must=filter_conditions
                )

        # Search in Qdrant with score_threshold and filter
        search_results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=models.NamedVector(name=vector_name, vector=query_vector),
            limit=limit,
            score_threshold=score_threshold,
            filter=filter_condition
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
                score=result.score
            )
            for result in search_results
        ] 