"""User profile endpoints — own profile and self-registration."""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.backend.core.auth import get_current_user
from src.backend.core.supabase_client import get_supabase_client
from src.backend.services.user_service import UserService

router = APIRouter(prefix="/api", tags=["profile"])


def get_user_service() -> UserService:
    return UserService(supabase=get_supabase_client())


class ProfileResponse(BaseModel):
    user_id: str
    email: str
    approved: bool
    is_admin: bool


class RegisterRequest(BaseModel):
    token: Optional[str] = None
    user_id: Optional[str] = None


@router.get("/users/me", response_model=ProfileResponse)
def get_my_profile(
    user_id: Annotated[str, Depends(get_current_user)],
    svc: Annotated[UserService, Depends(get_user_service)],
) -> ProfileResponse:
    profile = svc.get_profile(user_id=user_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profile not found. Contact the administrator.",
        )
    return ProfileResponse(**profile)


@router.post("/users/register", status_code=status.HTTP_201_CREATED)
def register_user(
    body: RegisterRequest,
    svc: Annotated[UserService, Depends(get_user_service)],
) -> dict:
    client = get_supabase_client()

    if body.token:
        try:
            response = client.auth.get_user(body.token)
            user = response.user
            if user is None:
                raise HTTPException(status_code=401, detail="Invalid token.")
        except Exception as exc:
            raise HTTPException(
                status_code=401, detail="Invalid or expired token."
            ) from exc
    elif body.user_id:
        # Email confirmation pending — verify the user actually exists via service role
        try:
            response = client.auth.admin.get_user_by_id(body.user_id)
            user = response.user
            if user is None:
                raise HTTPException(status_code=401, detail="User not found.")
        except Exception as exc:
            raise HTTPException(
                status_code=401, detail="Could not verify user."
            ) from exc
    else:
        raise HTTPException(status_code=400, detail="Provide token or user_id.")

    try:
        svc.create_profile(user_id=user.id, email=user.email or "", approved=False)
    except Exception:
        pass  # profile already exists — idempotent

    return {"created": True}
