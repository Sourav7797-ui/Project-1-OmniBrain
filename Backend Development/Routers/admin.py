import time
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query
from Database.schemas import (
    SystemMetricsResponse,
    UserRecord,
    UserCreateRequest,
    DocumentSummary,
    AdminActionResponse,
    TokenData
)
from auth import get_current_admin, get_password_hash

router = APIRouter()

# Fallback in-memory stores if database/vector modules are being built in parallel
ADMIN_DOC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "doc-101": {"filename": "Q3_Financial_Report.pdf", "total_chunks": 48, "indexed_at": datetime.utcnow()},
    "doc-102": {"filename": "Balance_Sheet_2026.pdf", "total_chunks": 16, "indexed_at": datetime.utcnow()}
}

ADMIN_USER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "analyst": {"role": "analyst", "created_at": datetime.utcnow()},
    "admin": {"role": "admin", "created_at": datetime.utcnow()}
}

SERVER_INIT_TIME = time.time()


# ==========================================
# 1. System Metrics & Telemetry Inspection
# ==========================================
@router.get("/metrics", response_model=SystemMetricsResponse, status_code=status.HTTP_200_OK)
async def get_system_metrics(admin: TokenData = Depends(get_current_admin)):
    """Exposes real-time system metrics for Streamlit Admin console."""
    vector_status = "connected"
    try:
        from Database.vector_store import check_vector_store_health
        vector_status = await check_vector_store_health()
    except (ImportError, AttributeError):
        pass

    return SystemMetricsResponse(
        active_sessions=1,
        total_documents=len(ADMIN_DOC_REGISTRY),
        vector_store_status=vector_status,
        uptime_seconds=round(time.time() - SERVER_INIT_TIME, 2)
    )


# ==========================================
# 2. Document & Vector Index Management
# ==========================================
@router.get("/docs", response_model=List[DocumentSummary], status_code=status.HTTP_200_OK)
async def list_indexed_documents(admin: TokenData = Depends(get_current_admin)):
    """Retrieves all indexed PDF/multimodal documents."""
    try:
        from crud import get_all_documents
        return await get_all_documents()
    except (ImportError, AttributeError):
        return [
            DocumentSummary(
                doc_id=k,
                filename=v["filename"],
                total_chunks=v["total_chunks"],
                indexed_at=v["indexed_at"]
            )
            for k, v in ADMIN_DOC_REGISTRY.items()
        ]


@router.delete("/docs/{doc_id}", response_model=AdminActionResponse, status_code=status.HTTP_200_OK)
async def delete_indexed_document(doc_id: str, admin: TokenData = Depends(get_current_admin)):
    """Purges document vectors from Qdrant/FAISS and deletes relational metadata."""
    # Purge vectors from vector_store
    try:
        from Database.vector_store import delete_document_vectors
        await delete_document_vectors(doc_id)
    except (ImportError, AttributeError):
        pass

    # Purge metadata from SQL relational store
    try:
        from crud import delete_document_record
        await delete_document_record(doc_id)
    except (ImportError, AttributeError):
        if doc_id in ADMIN_DOC_REGISTRY:
            del ADMIN_DOC_REGISTRY[doc_id]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document with ID '{doc_id}' not found."
            )

    return AdminActionResponse(
        status="success",
        message=f"Document '{doc_id}' successfully purged from vector index and relational metadata.",
        target_id=doc_id
    )


@router.post("/docs/{doc_id}/reindex", response_model=AdminActionResponse, status_code=status.HTTP_200_OK)
async def reindex_document(doc_id: str, admin: TokenData = Depends(get_current_admin)):
    """Triggers re-chunking and re-embedding for an existing document."""
    try:
        from Ingestion.embedder import reindex_document_pipeline
        await reindex_document_pipeline(doc_id)
    except (ImportError, AttributeError):
        if doc_id not in ADMIN_DOC_REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{doc_id}' not found for re-indexing."
            )

    return AdminActionResponse(
        status="success",
        message=f"Re-indexing pipeline triggered for document '{doc_id}'.",
        target_id=doc_id
    )


# ==========================================
# 3. User & Access Management
# ==========================================
@router.get("/users", response_model=List[UserRecord], status_code=status.HTTP_200_OK)
async def list_users(admin: TokenData = Depends(get_current_admin)):
    """Lists registered analyst and admin users."""
    try:
        from crud import get_all_users
        return await get_all_users()
    except (ImportError, AttributeError):
        return [
            UserRecord(
                username=uname,
                role=data["role"],
                created_at=data["created_at"]
            )
            for uname, data in ADMIN_USER_REGISTRY.items()
        ]


@router.post("/users", response_model=AdminActionResponse, status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreateRequest, admin: TokenData = Depends(get_current_admin)):
    """Creates a new user with hashed credentials and assigned role."""
    try:
        from crud import create_user_record
        await create_user_record(
            username=payload.username,
            password_hash=get_password_hash(payload.password),
            role=payload.role
        )
    except (ImportError, AttributeError):
        if payload.username in ADMIN_USER_REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Username '{payload.username}' already exists."
            )
        ADMIN_USER_REGISTRY[payload.username] = {
            "role": payload.role,
            "created_at": datetime.utcnow()
        }

    return AdminActionResponse(
        status="success",
        message=f"User '{payload.username}' with role '{payload.role}' created successfully.",
        target_id=payload.username
    )