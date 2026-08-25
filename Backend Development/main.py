import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from Routers import admin, chat, upload
from auth import create_access_token
from Database.schemas import Token

START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("OmniBrain Backend is initializing services...")
    yield
    print("OmniBrain Backend is shutting down...")

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

@app.post("/api/v1/auth/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    
    if form_data.username == "admin" and form_data.password == "admin123":
        role = "admin"
    elif form_data.username == "analyst" and form_data.password == "analyst123":
        role = "analyst"
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": form_data.username, "role": role})
    return {"access_token": access_token, "token_type": "bearer"}

app.include_router(upload.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat & Agents"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "online",
        "system": "OmniBrain Core",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }