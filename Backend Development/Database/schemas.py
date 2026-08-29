from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    job_id: str = Field(..., description="Unique UUID tracking the ingestion job")
    filename: str = Field(..., description="Original name of the uploaded document")
    status: str = Field(default="queued", description="Initial job status")
    message: str = Field(..., description="Status summary message")

class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="queued | processing | completed | failed")
    progress: int = Field(default=0, ge=0, le=100, description="Completion percentage")
    filename: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Citation(BaseModel):
    source: str = Field(..., description="Document filename or identifier")
    page: Optional[int] = Field(None, description="Page number where evidence was retrieved")
    snippet: str = Field(..., description="Relevant text excerpt, table, or chart explanation")
    score: Optional[float] = Field(None, description="Vector similarity / retrieval score")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Analyst financial question or instruction")
    session_id: str = Field(..., description="Active chat session identifier")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters for vector search")


class ChatResponse(BaseModel):
    session_id: str
    memo: str = Field(..., description="Synthesized investment memo or analytical response")
    citations: List[Citation] = Field(default_factory=list, description="Grounding citations")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class MessageRecord(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str
    citations: Optional[List[Citation]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[MessageRecord]

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = "analyst"

class SystemMetricsResponse(BaseModel):
    active_sessions: int
    total_documents: int
    vector_store_status: str
    uptime_seconds: float

class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: str = Field(default="analyst", description="analyst | admin")


class UserRecord(BaseModel):
    username: str
    role: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    total_chunks: int
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


class AdminActionResponse(BaseModel):
    status: str
    message: str
    target_id: Optional[str] = None