from typing import Any, Mapping, Optional, Sequence, Union

from qdrant_client import QdrantClient, models

from keys import QDRANT_API_KEY, QDRANT_URL


class QdrantService:
    """Thin wrapper around QdrantClient with convenience methods."""

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        resolved_url = url or QDRANT_URL
        resolved_api_key = api_key or QDRANT_API_KEY
        if not resolved_url:
            raise ValueError("Qdrant URL is required")
        if not resolved_api_key:
            raise ValueError("Qdrant API key is required")
        self.client = QdrantClient(url=resolved_url, api_key=resolved_api_key, **kwargs)
        print(f"Connected to Qdrant at {resolved_url}")

    def collection_exists(self, collection_name: str) -> bool:
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

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

        return self.client.upsert(collection_name=collection_name, points=normalized, **kwargs)
