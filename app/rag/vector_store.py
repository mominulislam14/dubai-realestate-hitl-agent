"""
Local vector store and semantic retrieval using Qdrant and FastEmbed.
Provides idempotent ingestion and low-latency similarity search without external embedding APIs.
"""

import json
import uuid
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from app.core.config import settings

_qdrant_client: QdrantClient | None = None
_embedding_model: TextEmbedding | None = None


def get_qdrant_client() -> QdrantClient:
    """Return a singleton QdrantClient instance to avoid local storage lock conflicts."""
    global _qdrant_client
    if _qdrant_client is None:
        Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
    return _qdrant_client


def get_embedding_model() -> TextEmbedding:
    """Return a lazily initialized singleton FastEmbed model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
    return _embedding_model


def ingest_knowledge_base() -> int:
    """Load JSON knowledge base and perform idempotent upsert into Qdrant."""
    kb_path = Path(settings.KB_DATA_PATH)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    with kb_path.open("r", encoding="utf-8") as f:
        documents: list[dict[str, Any]] = json.load(f)

    if not documents:
        return 0

    client = get_qdrant_client()
    embedder = get_embedding_model()

    # Combine title and content for richer semantic embedding
    texts_to_embed = [f"{doc['title']}\n{doc['content']}" for doc in documents]
    vectors = [vec.tolist() for vec in embedder.embed(texts_to_embed)]
    vector_size = len(vectors[0])

    if not client.collection_exists(settings.QDRANT_COLLECTION):
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    points = []
    for doc, vector in zip(documents, vectors):
        # Deterministic UUID ensures idempotent upserts on repeated runs
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc["id"]))
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "doc_id": doc["id"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "content": doc["content"],
                },
            )
        )

    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
    return len(points)


def ensure_kb_ready() -> None:
    """Ensure the vector collection exists and is populated."""
    client = get_qdrant_client()
    if not client.collection_exists(settings.QDRANT_COLLECTION):
        ingest_knowledge_base()
        return

    info = client.get_collection(settings.QDRANT_COLLECTION)
    if (info.points_count or 0) == 0:
        ingest_knowledge_base()


def search_knowledge_base(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k semantically relevant documents for a given query."""
    ensure_kb_ready()
    client = get_qdrant_client()
    embedder = get_embedding_model()

    query_vector = list(embedder.embed([query]))[0].tolist()
    response = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
    )

    results: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "doc_id": payload.get("doc_id", ""),
                "title": payload.get("title", ""),
                "category": payload.get("category", ""),
                "content": payload.get("content", ""),
                "score": round(float(point.score), 4),
            }
        )
    return results