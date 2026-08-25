from fastapi import APIRouter, Depends
from Database.schemas import TokenData
from auth import get_current_user

router = APIRouter()

@router.get("/upload/test", summary="Verify upload auth")
async def test_upload_auth(user: TokenData = Depends(get_current_user)):
    return {"message": f"Upload access verified for {user.username} with role {user.role}"}