from typing import Any, Mapping, Optional, Sequence, Union

from qdrant_client import QdrantClient, models

from keys import QDRANT_API_KEY, QDRANT_URL


class QdrantService:
    """Thin wrapper around QdrantClient with convenience methods."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, logger=None, **kwargs):
        resolved_url = url or QDRANT_URL
        resolved_api_key = api_key or QDRANT_API_KEY
        if not resolved_url:
            raise ValueError("Qdrant URL is required")
        if not resolved_api_key:
            raise ValueError("Qdrant API key is required")
        self.client = QdrantClient(url=resolved_url, api_key=resolved_api_key, **kwargs)
        if logger:
            self.logger = logger
            self.logger.info(f"[QdrantService] Connected to Qdrant at {resolved_url}")

    def collection_exists(self, collection_name: str) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    
    def get_collections(self) -> Sequence[models.CollectionInfo]:
        """Get list of existing collections."""
        return self.client.get_collections().collections

    def get_collection_names(self) -> Sequence[str]:
        """Get list of existing collection names."""
        return [c.name for c in self.get_collections()]

    def ensure_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: models.Distance = models.Distance.COSINE,
        **kwargs,
    ) -> bool:
        """Create collection if missing. Returns True when created, False if already present."""
        if self.collection_exists(collection_name):
            return False
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
            **kwargs,
        )
        return True

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection if it exists."""
        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name=collection_name)

    def upsert_points(
        self,
        collection_name: str,
        points: Sequence[Union[models.PointStruct, Mapping[str, Any]]],
        **kwargs,
    ) -> models.UpdateResult:
        """Upsert points into a collection. Points can be PointStruct or dicts with id/vector[/payload]."""
        if not points:
            raise ValueError("points must not be empty")

        normalized = []
        for point in points:
            if isinstance(point, models.PointStruct):
                normalized.append(point)
            elif isinstance(point, Mapping):
                if "id" not in point or "vector" not in point:
                    raise ValueError("point mapping must include 'id' and 'vector'")
                normalized.append(models.PointStruct(**point))
            else:
                raise TypeError("points must be PointStruct or mapping with id/vector")
        if self.logger:
            self.logger.info(f"Upserting {len(normalized)} points into collection {collection_name}")
        return self.client.upsert(collection_name=collection_name, points=normalized, **kwargs)
    
    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 5,
    ) -> Sequence[models.ScoredPoint]:
        """Search for nearest neighbors in a collection."""

        nearest = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=limit
        ).points

        parsed_point_results = []
        for point in nearest:
            point_dict = {
                "id": point.id,
                "score": point.score,
                "original_chunk_id": point.id % 1000
            }
            for key, value in point.payload.items():
                point_dict[key] = value
            parsed_point_results.append(point_dict)

        if self.logger:
            self.logger.info(f"[QdrantService] Search in collection {collection_name} returned {len(parsed_point_results)} results.")
        return parsed_point_results
