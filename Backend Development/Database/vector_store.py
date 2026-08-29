"""
Async Qdrant Vector Store Service for OmniBrain.
Manages 'omnibrain_text' and 'omnibrain_images' collections, batch upserts,
payload filtering, and hybrid multi-modal similarity searches.
"""

import os
import uuid
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

# Load environment configuration
load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None) or None
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "False").lower() in ("true", "1", "t")

TEXT_COLLECTION = os.getenv("QDRANT_TEXT_COLLECTION", "omnibrain_text")
IMAGE_COLLECTION = os.getenv("QDRANT_IMAGE_COLLECTION", "omnibrain_images")

TEXT_VECTOR_SIZE = int(os.getenv("TEXT_VECTOR_SIZE", "768"))
IMAGE_VECTOR_SIZE = int(os.getenv("IMAGE_VECTOR_SIZE", "512"))


class QdrantVectorStore:
    """Async vector store client wrapper for text and image collections."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        grpc_port: int = QDRANT_GRPC_PORT,
        api_key: Optional[str] = QDRANT_API_KEY,
        prefer_grpc: bool = QDRANT_PREFER_GRPC,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        
        self.client = AsyncQdrantClient(
            host=self.host,
            port=self.port,
            grpc_port=grpc_port,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
            timeout=30.0,
        )
        self.text_collection = TEXT_COLLECTION
        self.image_collection = IMAGE_COLLECTION
        self.text_vector_size = TEXT_VECTOR_SIZE
        self.image_vector_size = IMAGE_VECTOR_SIZE

    async def init_collections(self) -> None:
        """
        Creates 'omnibrain_text' and 'omnibrain_images' collections
        if they do not already exist.
        """
        existing_collections_res = await self.client.get_collections()
        existing_names = [c.name for c in existing_collections_res.collections]

        # 1. Initialize Text Collection
        if self.text_collection not in existing_names:
            await self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=rest_models.VectorParams(
                    size=self.text_vector_size,
                    distance=rest_models.Distance.COSINE,
                ),
                optimizers_config=rest_models.OptimizersConfigDiff(
                    indexing_threshold=10000
                ),
            )
            # Create payload index for fast filtering
            await self._create_payload_indexes(self.text_collection)

        # 2. Initialize Image Collection
        if self.image_collection not in existing_names:
            await self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config=rest_models.VectorParams(
                    size=self.image_vector_size,
                    distance=rest_models.Distance.COSINE,
                ),
                optimizers_config=rest_models.OptimizersConfigDiff(
                    indexing_threshold=5000
                ),
            )
            await self._create_payload_indexes(self.image_collection)

    async def _create_payload_indexes(self, collection_name: str) -> None:
        """Create indexes for commonly filtered fields."""
        fields_to_index = ["document_id", "user_id", "status"]
        for field in fields_to_index:
            try:
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=rest_models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # Index may already exist

    # =========================================================================
    # UPSERT OPERATIONS
    # =========================================================================

    async def upsert_text_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Batch upsert text embeddings into the omnibrain_text collection.
        """
        if not vectors:
            return True

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            rest_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]

        await self.client.upsert(
            collection_name=self.text_collection,
            points=points,
            wait=True,
        )
        return True

    async def upsert_image_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Batch upsert image embeddings into the omnibrain_images collection.
        """
        if not vectors:
            return True

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = [
            rest_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]

        await self.client.upsert(
            collection_name=self.image_collection,
            points=points,
            wait=True,
        )
        return True

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    def _build_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[rest_models.Filter]:
        """Convert a python dictionary of key-values into Qdrant Filter models."""
        if not filter_dict:
            return None

        must_conditions = []
        for key, value in filter_dict.items():
            if value is None:
                continue
            if isinstance(value, list):
                must_conditions.append(
                    rest_models.FieldCondition(
                        key=key,
                        match=rest_models.MatchAny(any=value),
                    )
                )
            else:
                must_conditions.append(
                    rest_models.FieldCondition(
                        key=key,
                        match=rest_models.MatchValue(value=str(value)),
                    )
                )

        return rest_models.Filter(must=must_conditions) if must_conditions else None

    async def search_text(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search on the text collection.
        """
        qdrant_filter = self._build_filter(filter_dict)

        results = await self.client.search(
            collection_name=self.text_collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            with_payload=True,
        )

        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "payload": hit.payload or {},
            }
            for hit in results
        ]

    async def search_images(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search on the image collection.
        """
        qdrant_filter = self._build_filter(filter_dict)

        results = await self.client.search(
            collection_name=self.image_collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold if score_threshold > 0 else None,
            with_payload=True,
        )

        return [
            {
                "id": str(hit.id),
                "score": float(hit.score),
                "payload": hit.payload or {},
            }
            for hit in results
        ]

    async def hybrid_search(
        self,
        text_vector: Optional[List[float]] = None,
        image_vector: Optional[List[float]] = None,
        text_weight: float = 0.7,
        image_weight: float = 0.3,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Combines and ranks similarity results across both text and image modalities.
        """
        combined_results: Dict[str, Dict[str, Any]] = {}

        # 1. Search text if provided
        if text_vector:
            text_hits = await self.search_text(
                query_vector=text_vector,
                top_k=top_k * 2,
                filter_dict=filter_dict,
            )
            for hit in text_hits:
                doc_id = hit["payload"].get("document_id", hit["id"])
                combined_results[doc_id] = {
                    "id": hit["id"],
                    "document_id": doc_id,
                    "score": hit["score"] * text_weight,
                    "type": "text",
                    "payload": hit["payload"],
                }

        # 2. Search images if provided
        if image_vector:
            image_hits = await self.search_images(
                query_vector=image_vector,
                top_k=top_k * 2,
                filter_dict=filter_dict,
            )
            for hit in image_hits:
                doc_id = hit["payload"].get("document_id", hit["id"])
                if doc_id in combined_results:
                    combined_results[doc_id]["score"] += hit["score"] * image_weight
                else:
                    combined_results[doc_id] = {
                        "id": hit["id"],
                        "document_id": doc_id,
                        "score": hit["score"] * image_weight,
                        "type": "image",
                        "payload": hit["payload"],
                    }

        # 3. Sort by aggregated score descending
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        return sorted_results[:top_k]

    # =========================================================================
    # DELETION & HEALTH
    # =========================================================================

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> bool:
        """Delete all vector points associated with a specific document UUID."""
        await self.client.delete(
            collection_name=collection_name,
            points_selector=rest_models.FilterSelector(
                filter=rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="document_id",
                            match=rest_models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )
        return True

    async def health_check(self) -> bool:
        """Verify active connection to Qdrant cluster."""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close client connection."""
        await self.client.close()


# Global singleton instance
vector_store = QdrantVectorStore()