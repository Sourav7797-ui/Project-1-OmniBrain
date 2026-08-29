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
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    COMPLETED = "completed"
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
    email: Optional[EmailStr] = None
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = Field(default=None, max_length=100)
    role: UserRole = UserRole.USER
    is_active: bool = True


class UserCreate(BaseModel):
    """Payload for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, description="Plaintext password for registration")
    role: UserRole = UserRole.ANALYST
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None


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
    id: Optional[UUID] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    """Standard OAuth2 bearer token response."""
    access_token: str
    token_type: str = "bearer"


class TokenResponse(BaseModel):
    """Extended JWT Token response after successful authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Decoded JWT payload data."""
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = "analyst"

# ============================================================================
# 3. CITATION & RETRIEVAL DTOS
# ============================================================================

class Citation(BaseModel):
    """Source citation reference supporting an AI response."""
    id: Optional[UUID] = Field(default_factory=uuid4)
    source: Optional[str] = None
    document_id: Optional[UUID] = None
    document_title: Optional[str] = None
    source_type: CitationSource = CitationSource.DOCUMENT
    page: Optional[int] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    snippet: str = Field(..., description="Extracted context or text snippet")
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Relevance / similarity confidence score")
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
    id: Optional[UUID] = Field(default_factory=uuid4)
    session_id: Optional[str] = None
    role: MessageRole
    content: str
    image_url: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

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


class MessageRecord(BaseModel):
    """Lightweight record for conversational history."""
    role: str = Field(..., description="user | assistant")
    content: str
    citations: Optional[List[Citation]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HistoryResponse(BaseModel):
    """Response payload for chat history queries."""
    session_id: str
    messages: List[MessageRecord] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """High-level query request sent to the RAG pipeline."""
    query: str = Field(..., min_length=1, description="Analyst financial question or instruction")
    session_id: str = Field(..., description="Active chat session identifier")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for search")
    search_type: Optional[VectorSearchType] = VectorSearchType.HYBRID


class ChatResponse(BaseModel):
    """Structured response from the OmniBrain supervisor agent."""
    session_id: str
    memo: str = Field(..., description="Synthesized investment memo or analytical response")
    citations: List[Citation] = Field(default_factory=list, description="Grounding citations")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


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

# ============================================================================
# 5. DOCUMENT INGESTION & STATUS DTOS
# ============================================================================

class UploadResponse(BaseModel):
    """Immediate response after staging an upload."""
    job_id: str = Field(..., description="Unique UUID tracking the ingestion job")
    filename: str = Field(..., description="Original name of the uploaded document")
    status: str = Field(default="queued", description="Initial job status")
    message: str = Field(..., description="Status summary message")


class JobStatusResponse(BaseModel):
    """Status polling response for document processing."""
    job_id: str
    status: str = Field(..., description="queued | processing | completed | failed")
    progress: int = Field(default=0, ge=0, le=100, description="Completion percentage")
    filename: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


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


class DocumentSummary(BaseModel):
    """Document summary used in admin management lists."""
    doc_id: str
    filename: str
    total_chunks: int = 0
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


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

# ============================================================================
# 7. SYSTEM METRICS & ADMIN ACTIONS
# ============================================================================

class SystemMetricsResponse(BaseModel):
    """Metrics returned to Streamlit admin dashboard."""
    active_sessions: int
    total_documents: int
    vector_store_status: str
    uptime_seconds: float


class AdminActionResponse(BaseModel):
    """Confirmation payload for admin operations."""
    status: str
    message: str
    target_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """System health status contract."""
    status: str = "healthy"
    database_connected: bool
    qdrant_connected: bool
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)