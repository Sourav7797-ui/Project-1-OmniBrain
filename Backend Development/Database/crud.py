"""
Atomic, Reusable Async Database CRUD Helper Functions for OmniBrain.
Handles database operations for Users, Documents, Chat Sessions, Messages, and Audit Logs.
"""

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# Import models & schemas
try:
    from .Database.models import AuditLog, ChatMessage, ChatSession, Document, User
    from .Database.schemas import ChatSessionCreate, ChatSessionUpdate, UserCreate, UserUpdate
except ImportError:
    from Database.models import AuditLog, ChatMessage, ChatSession, Document, User
    from Database.schemas import ChatSessionCreate, ChatSessionUpdate, UserCreate, UserUpdate


# ============================================================================
# 1. USER CRUD OPERATIONS
# ============================================================================

async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    """Retrieve a single user by their primary UUID."""
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Retrieve a user by unique email address."""
    stmt = select(User).where(User.email == email.lower().strip())
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_user_by_username(db: AsyncSession, username: str) -> Optional[User]:
    """Retrieve a user by unique username."""
    stmt = select(User).where(User.username == username.strip())
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_users(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    active_only: bool = False,
) -> List[User]:
    """Retrieve a paginated list of users."""
    stmt = select(User)
    if active_only:
        stmt = stmt.where(User.is_active.is_(True))
    stmt = stmt.offset(skip).limit(limit).order_by(desc(User.created_at))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def create_user(
    db: AsyncSession,
    user_in: UserCreate,
    hashed_password: str,
) -> User:
    """Create and persist a new user record."""
    db_user = User(
        email=user_in.email.lower().strip(),
        username=user_in.username.strip(),
        full_name=user_in.full_name,
        hashed_password=hashed_password,
        role=user_in.role.value if hasattr(user_in.role, "value") else str(user_in.role),
        is_active=user_in.is_active,
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def update_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    user_in: UserUpdate,
    hashed_password: Optional[str] = None,
) -> Optional[User]:
    """Update existing user properties."""
    db_user = await get_user_by_id(db, user_id)
    if not db_user:
        return None

    update_data = user_in.model_dump(exclude_unset=True)
    if hashed_password:
        update_data["hashed_password"] = hashed_password
    if "role" in update_data and hasattr(update_data["role"], "value"):
        update_data["role"] = update_data["role"].value

    for field, value in update_data.items():
        setattr(db_user, field, value)

    await db.commit()
    await db.refresh(db_user)
    return db_user


async def delete_user(db: AsyncSession, user_id: uuid.UUID) -> bool:
    """Delete a user record and cascade associated resources."""
    stmt = delete(User).where(User.id == user_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


# ============================================================================
# 2. DOCUMENT CRUD OPERATIONS
# ============================================================================

async def get_document_by_id(db: AsyncSession, doc_id: uuid.UUID) -> Optional[Document]:
    """Retrieve document metadata by document UUID."""
    stmt = select(Document).where(Document.id == doc_id)
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_documents_by_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
) -> List[Document]:
    """Retrieve all documents owned by a user."""
    stmt = select(Document).where(Document.user_id == user_id)
    if status:
        stmt = stmt.where(Document.status == status)
    stmt = stmt.offset(skip).limit(limit).order_by(desc(Document.created_at))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def create_document(
    db: AsyncSession,
    user_id: uuid.UUID,
    title: str,
    file_type: str,
    file_size_bytes: int = 0,
    file_url: Optional[str] = None,
    page_count: Optional[int] = None,
    tags: Optional[List[str]] = None,
    meta_info: Optional[Dict[str, Any]] = None,
) -> Document:
    """Create a new document ingestion record."""
    db_doc = Document(
        user_id=user_id,
        title=title,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        file_url=file_url,
        page_count=page_count,
        status="pending",
        tags=tags or [],
        meta_info=meta_info or {},
        chunk_count=0,
    )
    db.add(db_doc)
    await db.commit()
    await db.refresh(db_doc)
    return db_doc


async def update_document_status(
    db: AsyncSession,
    doc_id: uuid.UUID,
    status: str,
    chunk_count: Optional[int] = None,
) -> Optional[Document]:
    """Update document ingestion status and chunk count."""
    db_doc = await get_document_by_id(db, doc_id)
    if not db_doc:
        return None

    db_doc.status = status
    if chunk_count is not None:
        db_doc.chunk_count = chunk_count

    await db.commit()
    await db.refresh(db_doc)
    return db_doc


async def delete_document(
    db: AsyncSession,
    doc_id: uuid.UUID,
    user_id: Optional[uuid.UUID] = None,
) -> bool:
    """Delete a document record."""
    stmt = delete(Document).where(Document.id == doc_id)
    if user_id:
        stmt = stmt.where(Document.user_id == user_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


# ============================================================================
# 3. CHAT SESSION CRUD OPERATIONS
# ============================================================================

async def get_chat_session_by_id(
    db: AsyncSession,
    session_id: uuid.UUID,
    include_messages: bool = False,
) -> Optional[ChatSession]:
    """Retrieve a chat session with optional message loading."""
    stmt = select(ChatSession).where(ChatSession.id == session_id)
    if include_messages:
        stmt = stmt.options(selectinload(ChatSession.messages))
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_chat_sessions_by_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    skip: int = 0,
    limit: int = 50,
    include_archived: bool = False,
) -> List[ChatSession]:
    """Retrieve chat sessions for a specific user."""
    stmt = select(ChatSession).where(ChatSession.user_id == user_id)
    if not include_archived:
        stmt = stmt.where(ChatSession.is_archived.is_(False))
    stmt = stmt.offset(skip).limit(limit).order_by(desc(ChatSession.updated_at))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def create_chat_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    session_in: ChatSessionCreate,
) -> ChatSession:
    """Create a new chat conversation session."""
    db_session = ChatSession(
        user_id=user_id,
        title=session_in.title or "New Conversation",
        system_prompt=session_in.system_prompt,
        meta_info=session_in.metadata or {},
        is_archived=False,
    )
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)
    return db_session


async def update_chat_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    session_in: ChatSessionUpdate,
) -> Optional[ChatSession]:
    """Update title or archive status of a chat session."""
    db_session = await get_chat_session_by_id(db, session_id)
    if not db_session:
        return None

    update_data = session_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_session, field, value)

    await db.commit()
    await db.refresh(db_session)
    return db_session


async def delete_chat_session(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """Delete a chat session and cascade all contained messages."""
    stmt = delete(ChatSession).where(ChatSession.id == session_id)
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount > 0


# ============================================================================
# 4. CHAT MESSAGE CRUD OPERATIONS
# ============================================================================

async def get_messages_by_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    limit: int = 100,
) -> List[ChatMessage]:
    """Retrieve messages in chronological order for a session."""
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def create_chat_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
    image_url: Optional[str] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    tokens_used: Optional[int] = None,
    meta_info: Optional[Dict[str, Any]] = None,
) -> ChatMessage:
    """Append a new message into a chat session."""
    db_msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        image_url=image_url,
        citations=citations or [],
        tokens_used=tokens_used,
        meta_info=meta_info or {},
    )
    db.add(db_msg)
    await db.commit()
    await db.refresh(db_msg)
    return db_msg


# ============================================================================
# 5. AUDIT LOG CRUD OPERATIONS
# ============================================================================

async def create_audit_log(
    db: AsyncSession,
    action: str,
    resource_type: str,
    user_id: Optional[uuid.UUID] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> AuditLog:
    """Record an audit log entry for system activities."""
    log_entry = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address=ip_address,
    )
    db.add(log_entry)
    await db.commit()
    await db.refresh(log_entry)
    return log_entry


async def get_audit_logs(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    user_id: Optional[uuid.UUID] = None,
    action: Optional[str] = None,
) -> List[AuditLog]:
    """Retrieve audit logs with optional filtering."""
    stmt = select(AuditLog)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    stmt = stmt.offset(skip).limit(limit).order_by(desc(AuditLog.timestamp))
    result = await db.execute(stmt)
    return list(result.scalars().all())