from fastapi import APIRouter, Depends
from Database.schemas import TokenData
from auth import get_current_admin

router = APIRouter()

@router.get("/test", summary="Verify admin RBAC")
async def test_admin_rbac(admin_user: TokenData = Depends(get_current_admin)):
    return {"message": f"Admin access granted for {admin_user.username}"}