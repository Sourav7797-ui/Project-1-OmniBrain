"""
Automated Pytest Unit Tests for Qdrant Vector Store Service.
Verifies vector upserts, filters, similarity searches, and hybrid ranking algorithms.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

try:
    from Backend.Database.vector_store import QdrantVectorStore
except ImportError:
    from Database.vector_store import QdrantVectorStore

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_vector_store():
    """Fixture providing a mock-backed QdrantVectorStore instance."""
    store = QdrantVectorStore(host="localhost", port=6333)
    store.client = AsyncMock()
    return store


# ============================================================================
# 1. INITIALIZATION & HEALTH TESTS
# ============================================================================

async def test_init_collections(mock_vector_store: QdrantVectorStore):
    """Verify that text and image collections are properly initialized."""
    mock_collections_res = MagicMock()
    mock_collections_res.collections = []
    mock_vector_store.client.get_collections.return_value = mock_collections_res

    await mock_vector_store.init_collections()

    # Should create both text and image collections
    assert mock_vector_store.client.create_collection.call_count == 2


async def test_health_check_success(mock_vector_store: QdrantVectorStore):
    """Verify health check returns True when Qdrant responds."""
    mock_vector_store.client.get_collections.return_value = MagicMock()
    healthy = await mock_vector_store.health_check()
    assert healthy is True


async def test_health_check_failure(mock_vector_store: QdrantVectorStore):
    """Verify health check returns False when Qdrant is unreachable."""
    mock_vector_store.client.get_collections.side_effect = Exception("Connection refused")
    healthy = await mock_vector_store.health_check()
    assert healthy is False


# ============================================================================
# 2. VECTOR UPSERT TESTS
# ============================================================================

async def test_upsert_text_vectors(mock_vector_store: QdrantVectorStore):
    """Test batch upsert of text vectors with metadata payloads."""
    vectors = [[0.1] * 768, [0.2] * 768]
    payloads = [
        {"document_id": "doc-1", "chunk_index": 0, "text": "First chunk"},
        {"document_id": "doc-1", "chunk_index": 1, "text": "Second chunk"},
    ]

    success = await mock_vector_store.upsert_text_vectors(vectors=vectors, payloads=payloads)
    assert success is True
    assert mock_vector_store.client.upsert.call_count == 1


async def test_upsert_image_vectors(mock_vector_store: QdrantVectorStore):
    """Test batch upsert of image vectors."""
    vectors = [[0.5] * 512]
    payloads = [{"image_id": "img-1", "url": "http://example.com/img.jpg"}]

    success = await mock_vector_store.upsert_image_vectors(vectors=vectors, payloads=payloads)
    assert success is True
    assert mock_vector_store.client.upsert.call_count == 1


# ============================================================================
# 3. SIMILARITY SEARCH TESTS
# ============================================================================

async def test_search_text(mock_vector_store: QdrantVectorStore):
    """Test text vector cosine similarity search."""
    hit1 = MagicMock(id="p1", score=0.92, payload={"text": "Matched chunk 1", "document_id": "doc-1"})
    hit2 = MagicMock(id="p2", score=0.85, payload={"text": "Matched chunk 2", "document_id": "doc-2"})
    mock_vector_store.client.search.return_value = [hit1, hit2]

    query_vec = [0.1] * 768
    results = await mock_vector_store.search_text(
        query_vector=query_vec,
        top_k=2,
        score_threshold=0.8,
        filter_dict={"document_id": "doc-1"}
    )

    assert len(results) == 2
    assert results[0]["score"] == 0.92
    assert results[0]["payload"]["text"] == "Matched chunk 1"


async def test_hybrid_search_ranking(mock_vector_store: QdrantVectorStore):
    """Test weighted multi-modal hybrid search fusion and ranking."""
    # Mock text search results
    text_hit = MagicMock(id="doc-1", score=0.9, payload={"document_id": "doc-1", "content": "Text content"})
    # Mock image search results
    image_hit = MagicMock(id="doc-1", score=0.8, payload={"document_id": "doc-1", "content": "Image diagram"})

    async def mock_search(collection_name, query_vector, **kwargs):
        if collection_name == mock_vector_store.text_collection:
            return [text_hit]
        return [image_hit]

    mock_vector_store.client.search.side_effect = mock_search

    results = await mock_vector_store.hybrid_search(
        text_vector=[0.1] * 768,
        image_vector=[0.2] * 512,
        text_weight=0.7,
        image_weight=0.3,
        top_k=5,
    )

    assert len(results) == 1
    # Expected weighted score: (0.9 * 0.7) + (0.8 * 0.3) = 0.63 + 0.24 = 0.87
    assert round(results[0]["score"], 2) == 0.87
    assert results[0]["document_id"] == "doc-1"


# ============================================================================
# 4. DELETION TESTS
# ============================================================================

async def test_delete_by_document_id(mock_vector_store: QdrantVectorStore):
    """Test deleting points associated with a specific document UUID."""
    success = await mock_vector_store.delete_by_document_id(
        collection_name="omnibrain_text",
        document_id="doc-12345"
    )
    assert success is True
    assert mock_vector_store.client.delete.call_count == 1