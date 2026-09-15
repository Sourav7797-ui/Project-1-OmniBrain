import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    role: Optional[str] = "user"

@router.post("/login")
def login(creds: LoginRequest):
    if not creds.username or not creds.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required."
        )
    return {
        "access_token": f"jwt-{uuid.uuid4().hex[:16]}",
        "token_type": "bearer",
        "user": {
            "username": creds.username,
            "role": "admin" if creds.username == "admin" else "user"
        }
    }

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: RegisterRequest):
    if not user.username or not user.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password are required."
        )
    return {
        "status": "success",
        "message": f"User '{user.username}' registered successfully.",
        "user": {
            "username": user.username,
            "email": user.email,
            "role": user.role
        }
    }

from Database.schemas import TokenData, UserRole

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "omnibrain-dev-insecure-secret-key-32bytes")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Points to the unified API auth route
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a raw password against its stored bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generates a secure bcrypt hash for passwords."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encodes user identity, IDs, and role claims into a signed JWT token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """Dependency: Decodes Bearer JWT, validates expiration, and extracts claims."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub") or payload.get("username")
        user_id: Optional[str] = payload.get("user_id")
        raw_role: Optional[Union[UserRole, str]] = payload.get("role", UserRole.USER)

        if username is None:
            raise credentials_exception

        # Parse string or Enum into valid UserRole instance
        parsed_role = UserRole.USER
        if isinstance(raw_role, UserRole):
            parsed_role = raw_role
        elif isinstance(raw_role, str):
            try:
                parsed_role = UserRole(raw_role.lower())
            except ValueError:
                parsed_role = UserRole.USER

        return TokenData(username=username, user_id=str(user_id) if user_id else None, role=parsed_role)
    except (JWTError, ValueError):
        raise credentials_exception


async def get_current_admin(
    current_user: TokenData = Depends(get_current_user),
) -> TokenData:
    """Dependency: Enforces administrator-only role access control."""
    user_role = current_user.role.value if isinstance(current_user.role, UserRole) else str(current_user.role)
    if user_role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Administrative privileges required",
        )
    return current_user