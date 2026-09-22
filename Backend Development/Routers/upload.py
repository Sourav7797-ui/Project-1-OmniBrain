import os
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status, Header

from Database import async_session_factory, Document
from Database.schemas import UploadResponse, JobStatusResponse, TokenData
from auth import get_current_user

router = APIRouter()

STAGING_DIR = Path("staging_uploads")
STAGING_DIR.mkdir(parents=True, exist_ok=True)

JOB_REGISTRY: Dict[str, Dict[str, Any]] = {}

ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpg",
    "text/plain": ".txt",
}

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


async def _resolve_user_optional(authorization: Optional[str] = Header(None)) -> TokenData:
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            return await get_current_user(token=token)
        except HTTPException:
            pass
    return TokenData(username="guest_user", role="user")


async def execute_ingestion_pipeline(job_id: str, file_path: Path, filename: str, username: str):
    """Parses document, stores chunks in Qdrant, and logs record in SQLite."""
    try:
        JOB_REGISTRY[job_id]["status"] = "processing"
        JOB_REGISTRY[job_id]["progress"] = 20

        # 1. Extract text
        from Ingestion.pdf_extractor import extract_text_from_pdf, chunk_text
        if file_path.suffix.lower() == ".pdf":
            raw_text = extract_text_from_pdf(str(file_path))
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

        JOB_REGISTRY[job_id]["progress"] = 50

        # 2. Chunk text
        chunks = chunk_text(raw_text)
        JOB_REGISTRY[job_id]["progress"] = 70

        # 3. Vectorize and persist in Qdrant
        from Ingestion.embedder import embed_and_store_chunks
        # Store under both the actual user and guest_user to ensure search accessibility
        embed_and_store_chunks(
            chunks=chunks,
            user_id=username,
            parent_asset_id=filename,
            asset_type="pdf",
            extra_metadata={"filename": filename}
        )
        if username != "guest_user":
            embed_and_store_chunks(
                chunks=chunks,
                user_id="guest_user",
                parent_asset_id=filename,
                asset_type="pdf",
                extra_metadata={"filename": filename}
            )

        JOB_REGISTRY[job_id]["progress"] = 90

        # 4. Insert Document record into SQL database so Admin sees it
        try:
            async with async_session_factory() as session:
                async with session.begin():
                    doc_entry = Document(
                        filename=filename,
                        content_type="application/pdf",
                        file_size=file_path.stat().st_size,
                        chunks_count=len(chunks),
                        status="indexed"
                    )
                    session.add(doc_entry)
        except Exception as db_err:
            print(f"Warning: Could not save to SQL Document table: {db_err}")

        JOB_REGISTRY[job_id]["progress"] = 100
        JOB_REGISTRY[job_id]["status"] = "completed"
        JOB_REGISTRY[job_id]["completed_at"] = datetime.now(timezone.utc)

    except Exception as exc:
        JOB_REGISTRY[job_id]["status"] = "failed"
        JOB_REGISTRY[job_id]["error"] = str(exc)
        JOB_REGISTRY[job_id]["completed_at"] = datetime.now(timezone.utc)


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None),
):
    current_user = await _resolve_user_optional(authorization)

    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file.content_type not in ALLOWED_MIME_TYPES and file_ext not in ALLOWED_MIME_TYPES.values():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported format '{file.content_type}'."
        )

    job_id = str(uuid.uuid4())
    clean_filename = Path(file.filename).name if file.filename else f"doc_{job_id}.pdf"
    staged_file_path = STAGING_DIR / f"{job_id}_{clean_filename}"

    file_size = 0
    try:
        with staged_file_path.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File exceeds maximum size."
                    )
                buffer.write(chunk)
    finally:
        await file.close()

    JOB_REGISTRY[job_id] = {
        "job_id": job_id,
        "filename": clean_filename,
        "status": "queued",
        "progress": 0,
        "uploaded_by": current_user.username,
        "file_path": str(staged_file_path),
        "error": None,
        "created_at": datetime.now(timezone.utc)
    }

    background_tasks.add_task(
        execute_ingestion_pipeline,
        job_id,
        staged_file_path,
        clean_filename,
        current_user.username
    )

    return UploadResponse(
        job_id=job_id,
        filename=clean_filename,
        status="success",
        total_documents=1,
        files=[{
            "filename": clean_filename,
            "job_id": job_id,
            "status": "queued",
            "size_bytes": file_size
        }],
        message="Document queued for ingestion."
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_ingestion_status(job_id: str):
    job = JOB_REGISTRY.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        filename=job.get("filename"),
        error=job.get("error"),
        created_at=job.get("created_at", datetime.now(timezone.utc))
    )