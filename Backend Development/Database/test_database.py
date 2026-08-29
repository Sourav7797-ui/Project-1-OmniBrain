"""
Automated Pytest Unit Tests for Relational Database & CRUD Operations.
Uses an in-memory SQLite async engine to test models, relationships, and queries.
"""

import uuid
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Import Base, Models, and CRUD helpers
try:
    from Backend.Database.database import Base
    from Backend.Database.models import AuditLog, ChatMessage, ChatSession, Document, User
    from Backend.Database.schemas import ChatSessionCreate, ChatSessionUpdate, UserCreate, UserRole, UserUpdate
    from Backend import crud
except ImportError:
    from Database.database import Base
    from Database.models import AuditLog, ChatMessage, ChatSession, Document, User
    from Database.schemas import ChatSessionCreate, ChatSessionUpdate, UserCreate, UserRole, UserUpdate
    import crud

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

# In-memory async SQLite engine for isolated testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="function")
async def test_db_session():
    """Fixture that initializes fresh database tables and yields an async session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# ============================================================================
# 1. USER TESTS
# ============================================================================

async def test_create_and_get_user(test_db_session: AsyncSession):
    """Test user registration and retrieval by ID, email, and username."""
    user_in = UserCreate(
        email="testuser@omnibrain.ai",
        username="testuser",
        full_name="Test User",
        password="securepassword123",
        role=UserRole.USER,
    )
    user = await crud.create_user(test_db_session, user_in, hashed_password="hashed_securepassword123")

    assert user.id is not None
    assert user.email == "testuser@omnibrain.ai"
    assert user.username == "testuser"

    # Fetch by ID
    fetched_by_id = await crud.get_user_by_id(test_db_session, user.id)
    assert fetched_by_id is not None
    assert fetched_by_id.id == user.id

    # Fetch by Email
    fetched_by_email = await crud.get_user_by_email(test_db_session, "testuser@omnibrain.ai")
    assert fetched_by_email is not None
    assert fetched_by_email.email == "testuser@omnibrain.ai"

    # Fetch by Username
    fetched_by_username = await crud.get_user_by_username(test_db_session, "testuser")
    assert fetched_by_username is not None
    assert fetched_by_username.username == "testuser"


async def test_update_and_delete_user(test_db_session: AsyncSession):
    """Test updating user attributes and deletion."""
    user_in = UserCreate(
        email="update@omnibrain.ai",
        username="updateuser",
        full_name="Old Name",
        password="password123",
    )
    user = await crud.create_user(test_db_session, user_in, hashed_password="hashed_pw")

    # Update
    updated = await crud.update_user(
        test_db_session,
        user.id,
        UserUpdate(full_name="New Name", role=UserRole.ADMIN)
    )
    assert updated.full_name == "New Name"
    assert updated.role == "admin"

    # Delete
    deleted = await crud.delete_user(test_db_session, user.id)
    assert deleted is True

    # Verify gone
    missing = await crud.get_user_by_id(test_db_session, user.id)
    assert missing is None


# ============================================================================
# 2. DOCUMENT TESTS
# ============================================================================

async def test_document_lifecycle(test_db_session: AsyncSession):
    """Test creating document, updating status, and retrieving documents."""
    user_in = UserCreate(email="docuser@omnibrain.ai", username="docuser", password="password123")
    user = await crud.create_user(test_db_session, user_in, hashed_password="hashed_pw")

    doc = await crud.create_document(
        test_db_session,
        user_id=user.id,
        title="Research Paper.pdf",
        file_type="application/pdf",
        file_size_bytes=102400,
        page_count=12,
        tags=["ai", "rag"],
        meta_info={"author": "OmniBrain Team"},
    )

    assert doc.id is not None
    assert doc.status == "pending"
    assert doc.tags == ["ai", "rag"]

    # Update status
    updated_doc = await crud.update_document_status(test_db_session, doc.id, status="indexed", chunk_count=24)
    assert updated_doc.status == "indexed"
    assert updated_doc.chunk_count == 24

    # List by user
    docs = await crud.get_documents_by_user(test_db_session, user.id)
    assert len(docs) == 1
    assert docs[0].title == "Research Paper.pdf"


# ============================================================================
# 3. CHAT SESSION & MESSAGE TESTS
# ============================================================================

async def test_chat_session_and_messages(test_db_session: AsyncSession):
    """Test multi-turn chat sessions and message ordering."""
    user_in = UserCreate(email="chatuser@omnibrain.ai", username="chatuser", password="password123")
    user = await crud.create_user(test_db_session, user_in, hashed_password="hashed_pw")

    session_in = ChatSessionCreate(title="RAG Architecture Discussion", system_prompt="You are a helpful assistant.")
    session = await crud.create_chat_session(test_db_session, user.id, session_in)
    assert session.id is not None
    assert session.title == "RAG Architecture Discussion"

    # Add messages
    msg1 = await crud.create_chat_message(
        test_db_session,
        session_id=session.id,
        role="user",
        content="How does Qdrant handle hybrid search?",
    )
    msg2 = await crud.create_chat_message(
        test_db_session,
        session_id=session.id,
        role="assistant",
        content="Qdrant combines dense vector similarity with sparse or multi-modal scoring.",
        citations=[{"doc_id": "123", "snippet": "Qdrant hybrid search documentation"}],
        tokens_used=42,
    )

    assert msg1.id is not None
    assert msg2.citations[0]["doc_id"] == "123"

    # Retrieve message history
    messages = await crud.get_messages_by_session(test_db_session, session.id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


# ============================================================================
# 4. AUDIT LOG TESTS
# ============================================================================

async def test_audit_logs(test_db_session: AsyncSession):
    """Test recording and querying audit logs."""
    log = await crud.create_audit_log(
        test_db_session,
        action="user_login",
        resource_type="auth",
        resource_id="session_xyz",
        details={"ip": "127.0.0.1"},
    )
    assert log.id is not None
    assert log.action == "user_login"

    logs = await crud.get_audit_logs(test_db_session, action="user_login")
    assert len(logs) >= 1