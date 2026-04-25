"""Admin routes — user management (create, list, approve/revoke)."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.backend.core.auth import get_current_user
from src.backend.core.supabase_client import get_supabase_client
from src.backend.services.user_service import UserService

router = APIRouter(prefix="/api/admin", tags=["admin"])


def get_user_service() -> UserService:
    return UserService(supabase=get_supabase_client())


def get_admin_user(
    user_id: Annotated[str, Depends(get_current_user)],
    svc: Annotated[UserService, Depends(get_user_service)],
) -> str:
    """Dependency — raises 403 unless the caller has is_admin = true."""
    profile = svc.get_profile(user_id=user_id)
    if profile is None or not profile.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user_id


class UserResponse(BaseModel):
    user_id: str
    email: str
    approved: bool
    is_admin: bool


class CreateUserRequest(BaseModel):
    email: str
    password: str


class PatchUserRequest(BaseModel):
    approved: bool


class SuccessResponse(BaseModel):
    success: bool


@router.get("/users", response_model=list[UserResponse])
def list_users(
    _: Annotated[str, Depends(get_admin_user)],
    svc: Annotated[UserService, Depends(get_user_service)],
) -> list[UserResponse]:
    """Return all registered users."""
    return [UserResponse(**u) for u in svc.list_users()]


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    body: CreateUserRequest,
    _: Annotated[str, Depends(get_admin_user)],
    svc: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """Create a new user account (admin-invited, immediately approved)."""
    client = get_supabase_client()
    try:
        result = client.auth.admin.create_user(
            {"email": body.email, "password": body.password, "email_confirm": True}
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create user: {exc}",
        )

    new_user = result.user
    if not new_user:
        raise HTTPException(status_code=500, detail="User creation returned no user")

    svc.create_profile(
        user_id=str(new_user.id),
        email=body.email,
        approved=True,
    )
    return UserResponse(
        user_id=str(new_user.id),
        email=body.email,
        approved=True,
        is_admin=False,
    )


@router.patch("/users/{user_id}", response_model=SuccessResponse)
def patch_user(
    user_id: str,
    body: PatchUserRequest,
    _: Annotated[str, Depends(get_admin_user)],
    svc: Annotated[UserService, Depends(get_user_service)],
) -> SuccessResponse:
    """Approve or revoke a user's access."""
    svc.set_approved(user_id=user_id, approved=body.approved)
    return SuccessResponse(success=True)
