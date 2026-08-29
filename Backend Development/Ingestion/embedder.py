import os
import uuid
import ollama
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, PointIdsList
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import PayloadSchemaType

load_dotenv()

# Qdrant Cloud Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pdf_knowledge_base"

# Embedding Model Config (nomic-embed-text generates 768-dimensional vectors)
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_DIMENSION = 768

# Initialize Qdrant Cloud Client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

def init_qdrant_collection():
    collections = [col.name for col in qdrant_client.get_collections().collections]
    
    if COLLECTION_NAME not in collections:
        # 1. Create Collection
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIMENSION, distance=Distance.COSINE)
        )
        
        # 2. Create Payload Index for user_id filtering
        qdrant_client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print(f"✅ Created Qdrant collection & indexed user_id field: {COLLECTION_NAME}")

# Initialize on module import
init_qdrant_collection()


def get_embedding(text: str) -> List[float]:
    """Generates a vector embedding with strict token length protection."""
    words = text.split()
    safe_text = " ".join(words[:400]) if len(words) > 400 else text

    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=safe_text
    )
    return response["embedding"]


def embed_and_store_chunks(
    chunks: List[str],
    user_id: str,  # Pass the unique User ID
    parent_asset_id: str,
    asset_type: str,
    cloudinary_public_id: str = "",
    cloudinary_url: str = "",
    extra_metadata: Dict[str, Any] = None
) -> List[str]:
    points = []
    stored_ids = []

    for idx, chunk in enumerate(chunks, start=1):
        point_id = str(uuid.uuid4())
        vector = get_embedding(chunk)

        # Attach user_id directly to payload
        payload = {
            "user_id": user_id,
            "chunk_text": chunk,
            "parent_asset_id": parent_asset_id,
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
            points=points
        )

    return stored_ids


def search_user_knowledge_base(
    user_id: str, 
    query: str, 
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieves semantic matches strictly belonging to the specified user_id.
    """
    query_vector = get_embedding(query)
    
    # Restrict search solely to this user's vectors
    user_filter = Filter(
        must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        ]
    )

    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=user_filter,  # Enforces multi-tenancy privacy
        limit=limit
    ).points

    return [
        {
            "score": hit.score,
            "text": hit.payload.get("chunk_text"),
            "cloudinary_url": hit.payload.get("cloudinary_url"),
            "asset_type": hit.payload.get("type"),
            "user_id": hit.payload.get("user_id")
        }
        for hit in search_results
    ]