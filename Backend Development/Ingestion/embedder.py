"""
OmniBrain Document Embedding and Ingestion Service.
Supports batch embedding with Ollama, multi-tenant Qdrant storage,
lazy initialization, and scoped semantic search.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL") or f"http://{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("QDRANT_TEXT_COLLECTION", "omnibrain_text")

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
VECTOR_DIMENSION = int(os.getenv("TEXT_VECTOR_SIZE", "768"))
MAX_WORDS_PER_CHUNK = 400

# Client Singleton
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=15.0
)

_collection_initialized = False


def init_qdrant_collection() -> None:
    """Lazily verifies and initializes collection and payload indices."""
    global _collection_initialized
    if _collection_initialized:
        return

    try:
        collections = [col.name for col in qdrant_client.get_collections().collections]

        if COLLECTION_NAME not in collections:
            # 1. Create Collection
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
            )

            # 2. Create Payload Indexes for multi-tenancy & doc lifecycle
            for field in ["user_id", "parent_asset_id", "type"]:
                try:
                    qdrant_client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    pass

            logger.info(f"Created Qdrant collection & indices: {COLLECTION_NAME}")
        _collection_initialized = True

    except Exception as exc:
        logger.warning(f"Could not connect to Qdrant during initialization: {exc}")


def _truncate_text(text: str) -> str:
    """Clamps words to stay well within embedding token budgets."""
    words = text.split()
    return " ".join(words[:MAX_WORDS_PER_CHUNK]) if len(words) > MAX_WORDS_PER_CHUNK else text


def get_embedding(text: str) -> List[float]:
    """Generates a single vector embedding."""
    safe_text = _truncate_text(text)
    try:
        # Use modern ollama.embed API
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=safe_text
        )
        return response["embeddings"][0]
    except AttributeError:
        # Fallback for older ollama package versions
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=safe_text
        )
        return response["embedding"]


def get_batch_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for multiple texts in a single round-trip."""
    if not texts:
        return []

    safe_texts = [_truncate_text(t) for t in texts]
    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=safe_texts
        )
        return response["embeddings"]
    except Exception:
        # Fallback to serial processing if the batch route is unsupported
        return [get_embedding(t) for t in safe_texts]


def embed_and_store_chunks(
    chunks: List[str],
    user_id: str,
    parent_asset_id: str,
    asset_type: str,
    cloudinary_public_id: str = "",
    cloudinary_url: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Batches embedding requests and persists indexed points to Qdrant.
    """
    if not chunks:
        return []

    init_qdrant_collection()

    # Generate all embeddings in a single batch request
    vectors = get_batch_embeddings(chunks)

    points: List[PointStruct] = []
    stored_ids: List[str] = []

    for idx, (chunk, vector) in enumerate(zip(chunks, vectors), start=1):
        point_id = str(uuid.uuid4())
        payload = {
            "user_id": str(user_id),
            "chunk_text": chunk,
            "parent_asset_id": str(parent_asset_id),
            "document_id": str(parent_asset_id),  # Cross-compatibility alias
            "type": asset_type,
            "chunk_index": idx,
            "cloudinary_public_id": cloudinary_public_id,
            "cloudinary_url": cloudinary_url
        }

        if extra_metadata:
            payload.update(extra_metadata)

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
        stored_ids.append(point_id)

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

    return stored_ids


def search_user_knowledge_base(
    user_id: str,
    query: str,
    limit: int = 5,
    score_threshold: Optional[float] = 0.0
) -> List[Dict[str, Any]]:
    """
    Executes scoped vector search enforcing user-level isolation.
    """
    init_qdrant_collection()
    query_vector = get_embedding(query)

    user_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=str(user_id))
            )
        ]
    )

    try:
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=user_filter,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None
        ).points

        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "text": hit.payload.get("chunk_text", ""),
                "cloudinary_url": hit.payload.get("cloudinary_url", ""),
                "asset_type": hit.payload.get("type", ""),
                "user_id": hit.payload.get("user_id", ""),
                "parent_asset_id": hit.payload.get("parent_asset_id", ""),
                "metadata": hit.payload
            }
            for hit in search_results
        ]
    except Exception as exc:
        logger.error(f"Failed to query knowledge base: {exc}")
        return []