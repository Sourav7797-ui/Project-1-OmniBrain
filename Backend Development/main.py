import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from Database import init_db, close_db
from Routers import auth, admin, chat, upload

logger = logging.getLogger("omnibrain.main")
START_TIME = time.time()


class GuardrailViolationException(Exception):
    def __init__(self, message: str, violation_type: str = "safety_policy_violation"):
        self.message = message
        self.violation_type = violation_type
        super().__init__(self.message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("OmniBrain Backend is initializing services...")
    try:
        # Initialize SQL tables on startup
        await init_db()
        print("Database tables verified and initialized.")
    except Exception as exc:
        print(f"Database initialization warning: {exc}")
    
    yield
    
    print("OmniBrain Backend is shutting down...")
    try:
        await close_db()
    except Exception as exc:
        print(f"Error closing DB sessions: {exc}")


app = FastAPI(
    title="OmniBrain API",
    description="Agentic Multi-Modal RAG Orchestrator Backend",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def trace_and_telemetry_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id", str(uuid.uuid4()))
    request.state.trace_id = trace_id

    start_time = time.time()
    response = await call_next(request)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    response.headers["x-trace-id"] = trace_id
    response.headers["x-latency-ms"] = str(latency_ms)
    return response


@app.exception_handler(GuardrailViolationException)
async def handle_guardrail_violation(request: Request, exc: GuardrailViolationException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "GuardrailViolation",
            "violation_type": exc.violation_type,
            "message": exc.message,
            "trace_id": getattr(request.state, "trace_id", None)
        }
    )


# ----------------------------------------------------------------------
# API Routers Registration
# ----------------------------------------------------------------------
# Auth: /api/v1/auth/login, /api/v1/auth/register
app.include_router(auth.router, tags=["Authentication"])

# Ingestion: /api/v1/upload
app.include_router(upload.router, prefix="/api/v1", tags=["Ingestion"])

# Chat & Agents: /api/v1/chat
app.include_router(chat.router, prefix="/api/v1", tags=["Chat & Agents"])

# Admin: /api/v1/admin/*
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "online",
        "system": "OmniBrain Core",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }