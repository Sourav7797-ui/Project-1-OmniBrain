"""
OmniBrain Database and Vector Store Module.
Exposes database models, session factories, and system-wide Pydantic schemas/DTOs.
"""

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