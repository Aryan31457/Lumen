from functools import lru_cache
from qdrant_client import QdrantClient, models
from .config import settings


@lru_cache
def client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def ensure_collections() -> None:
    c = client()
    if not c.collection_exists("faces"):
        c.create_collection("faces", vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))
        c.create_payload_index("faces", "person_id", models.PayloadSchemaType.INTEGER)
    if not c.collection_exists("photos"):
        c.create_collection("photos", vectors_config={
            "activity": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "clip": models.VectorParams(size=512, distance=models.Distance.COSINE),
        })
        for field, kind in (("person_ids", models.PayloadSchemaType.INTEGER), ("taken_at_ts", models.PayloadSchemaType.FLOAT), ("tags", models.PayloadSchemaType.KEYWORD)):
            c.create_payload_index("photos", field, kind)
