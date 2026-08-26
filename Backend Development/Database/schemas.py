"""
OmniBrain System-Wide Pydantic Data Contracts & DTOs.
Freezes contracts for Chat, Ingestion, Citations, Auth, and Vector Search.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ============================================================================
# 1. ENUMS
# ============================================================================

class UserRole(str, Enum):
    """User authorization roles."""
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"
    GUEST = "guest"


class MessageRole(str, Enum):
    """Chat message participant roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class DocumentStatus(str, Enum):
    """Ingestion and processing lifecycle statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class CitationSource(str, Enum):
    """Source origin of a retrieved citation."""
    DOCUMENT = "document"
    IMAGE = "image"
    WEB = "web"
    DATABASE = "database"


class VectorSearchType(str, Enum):
    """Vector similarity search modalities."""
    TEXT = "text"
    IMAGE = "image"
    HYBRID = "hybrid"


# ============================================================================
# 2. AUTHENTICATION & USER SCHEMAS
# ============================================================================

class UserBase(BaseModel):
    """Base user properties."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = Field(default=None, max_length=100)
    role: UserRole = UserRole.USER
    is_active: bool = True


class UserCreate(UserBase):
    """Payload for user registration."""
    password: str = Field(..., min_length=8, description="Plaintext password for registration")


class UserLogin(BaseModel):
    """Payload for user authentication."""
    username_or_email: str
    password: str


class UserUpdate(BaseModel):
    """Payload for updating user profile."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None


class UserResponse(UserBase):
    """User entity returned to clients."""
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """JWT Token response after successful authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Decoded JWT payload data."""
    user_id: Optional[str] = None
    role: Optional[UserRole] = None


# ============================================================================
# 3. CITATION & RETRIEVAL DTOS
# ============================================================================

class Citation(BaseModel):
    """Source citation reference supporting an AI response."""
    id: UUID = Field(default_factory=uuid4)
    document_id: Optional[UUID] = None
    document_title: Optional[str] = None
    source_type: CitationSource = CitationSource.DOCUMENT
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    snippet: str = Field(..., description="Extracted context or text snippet")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance / similarity confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# 4. CHAT SESSION & MESSAGE DTOS
# ============================================================================

class ChatMessageCreate(BaseModel):
    """Payload for sending a new message in a session."""
    role: MessageRole = MessageRole.USER
    content: str = Field(..., min_length=1, description="Message text content")
    image_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatMessageResponse(BaseModel):
    """Message entity returned to clients."""
    id: UUID
    session_id: UUID
    role: MessageRole
    content: str
    image_url: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatSessionCreate(BaseModel):
    """Payload for creating a new chat conversation."""
    title: Optional[str] = Field(default="New Conversation", max_length=150)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSessionUpdate(BaseModel):
    """Payload for updating session metadata."""
    title: Optional[str] = Field(None, max_length=150)
    is_archived: Optional[bool] = None


class ChatSessionResponse(BaseModel):
    """Chat session details with message history metadata."""
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_archived: bool = False
    message_count: int = 0
    messages: List[ChatMessageResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ChatQueryRequest(BaseModel):
    """High-level multi-modal query request sent to the RAG pipeline."""
    session_id: Optional[UUID] = None
    query: str = Field(..., min_length=1, description="User's query prompt")
    image_url: Optional[str] = None
    search_type: VectorSearchType = VectorSearchType.HYBRID
    filter_tags: List[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)


class ChatQueryResponse(BaseModel):
    """Structured response from the OmniBrain AI engine."""
    session_id: UUID
    message: ChatMessageResponse
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: float
    model_name: str


# ============================================================================
# 5. DOCUMENT INGESTION DTOS
# ============================================================================

class DocumentMetadata(BaseModel):
    """Structured metadata associated with an ingested file."""
    file_name: str
    file_type: str
    file_size_bytes: int
    page_count: Optional[int] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Extracted text or image chunk ready for vectorization."""
    chunk_index: int
    content: str
    token_count: Optional[int] = None
    page_number: Optional[int] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentIngestionRequest(BaseModel):
    """Request contract for ingesting a document into the system."""
    title: str = Field(..., min_length=1, max_length=255)
    file_url: Optional[str] = None
    metadata: DocumentMetadata
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=512)


class DocumentIngestionResponse(BaseModel):
    """Response returned upon initiating document processing."""
    document_id: UUID
    title: str
    status: DocumentStatus
    total_chunks: int = 0
    message: str = "Document queued for processing"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentSummaryResponse(BaseModel):
    """Summary representation for document listings."""
    id: UUID
    user_id: UUID
    title: str
    status: DocumentStatus
    file_type: str
    file_size_bytes: int
    chunk_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# 6. VECTOR STORE & SEARCH DTOS
# ============================================================================

class VectorSearchFilter(BaseModel):
    """Filters applicable to Qdrant vector similarity searches."""
    document_id: Optional[UUID] = None
    tags: List[str] = Field(default_factory=list)
    user_id: Optional[UUID] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    extra_conditions: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchQuery(BaseModel):
    """Query payload executed against Qdrant vector collections."""
    collection_name: str
    vector: List[float] = Field(..., description="Query dense embedding vector")
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    filter: Optional[VectorSearchFilter] = None


class VectorSearchResult(BaseModel):
    """Individual match returned from Qdrant."""
    point_id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class VectorUpsertRequest(BaseModel):
    """Payload for batch upserting points into Qdrant."""
    collection_name: str
    point_ids: List[str]
    vectors: List[List[float]]
    payloads: List[Dict[str, Any]]


class VectorUpsertResponse(BaseModel):
    """Status result of a batch vector upsert operation."""
    collection_name: str
    upserted_count: int
    status: str = "success"


# ============================================================================
# 7. AUDIT LOGS & SYSTEM HEALTH
# ============================================================================

class AuditLogCreate(BaseModel):
    """Contract for logging system events."""
    user_id: Optional[UUID] = None
    action: str = Field(..., min_length=1, max_length=100)
    resource_type: str = Field(..., max_length=50)
    resource_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None


class AuditLogResponse(BaseModel):
    """Audit log entry representation."""
    id: UUID
    user_id: Optional[UUID] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)


class HealthCheckResponse(BaseModel):
    """System health status contract."""
    status: str = "healthy"
    database_connected: bool
    qdrant_connected: bool
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)