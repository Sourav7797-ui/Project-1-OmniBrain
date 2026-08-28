"""
OmniBrain Database and Vector Store Module.
Exposes database models, session factories, vector store, and system-wide Pydantic schemas/DTOs.
"""

from .database import (
    Base,
    engine,
    AsyncSessionLocal,
    get_db,
    init_db,
    close_db,
)
from .models import (
    User,
    Document,
    ChatSession,
    ChatMessage,
    AuditLog,
)
from .vector_store import (
    QdrantVectorStore,
    vector_store,
)
from .schemas import (
    # Auth & User Schemas
    UserRole,
    UserBase,
    UserCreate,
    UserLogin,
    UserUpdate,
    UserResponse,
    TokenResponse,
    TokenData,
    
    # Document Ingestion DTOs
    DocumentStatus,
    DocumentMetadata,
    DocumentChunk,
    DocumentIngestionRequest,
    DocumentIngestionResponse,
    DocumentSummaryResponse,
    
    # Citation DTOs
    CitationSource,
    Citation,
    
    # Chat DTOs
    MessageRole,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    ChatQueryRequest,
    ChatQueryResponse,
    
    # Vector Search DTOs
    VectorSearchType,
    VectorSearchFilter,
    VectorSearchQuery,
    VectorSearchResult,
    VectorUpsertRequest,
    VectorUpsertResponse,
    
    # Audit & Health
    AuditLogCreate,
    AuditLogResponse,
    HealthCheckResponse,
)

__all__ = [
    # Database core
    "Base",
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "init_db",
    "close_db",
    # Vector Store
    "QdrantVectorStore",
    "vector_store",
    # ORM Models
    "User",
    "Document",
    "ChatSession",
    "ChatMessage",
    "AuditLog",
    # Pydantic Schemas / DTOs
    "UserRole",
    "UserBase",
    "UserCreate",
    "UserLogin",
    "UserUpdate",
    "UserResponse",
    "TokenResponse",
    "TokenData",
    "DocumentStatus",
    "DocumentMetadata",
    "DocumentChunk",
    "DocumentIngestionRequest",
    "DocumentIngestionResponse",
    "DocumentSummaryResponse",
    "CitationSource",
    "Citation",
    "MessageRole",
    "ChatMessageCreate",
    "ChatMessageResponse",
    "ChatSessionCreate",
    "ChatSessionUpdate",
    "ChatSessionResponse",
    "ChatQueryRequest",
    "ChatQueryResponse",
    "VectorSearchType",
    "VectorSearchFilter",
    "VectorSearchQuery",
    "VectorSearchResult",
    "VectorUpsertRequest",
    "VectorUpsertResponse",
    "AuditLogCreate",
    "AuditLogResponse",
    "HealthCheckResponse",
]