"""API routes for managing user Gemini API keys."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.backend.core.auth import get_current_user
from src.backend.core.encryption import EncryptionService
from src.backend.core.supabase_client import get_supabase_client
from src.backend.services.api_key_service import ApiKeyService

router = APIRouter(prefix="/api/keys", tags=["keys"])


def get_api_key_service() -> ApiKeyService:
    fernet_key = EncryptionService.key_from_env("ENCRYPTION_KEY")
    return ApiKeyService(
        supabase=get_supabase_client(),
        encryption=EncryptionService(fernet_key=fernet_key),
    )


class GeminiKeyStatus(BaseModel):
    has_key: bool


class GeminiKeyRequest(BaseModel):
    key: str


class SuccessResponse(BaseModel):
    success: bool


@router.get("/gemini", response_model=GeminiKeyStatus)
async def get_gemini_key_status(
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> GeminiKeyStatus:
    """Return whether the authenticated user has a Gemini key stored."""
    key = service.get_user_key(user_id=user_id)
    return GeminiKeyStatus(has_key=key is not None)


@router.put("/gemini", response_model=SuccessResponse)
async def save_gemini_key(
    body: GeminiKeyRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> SuccessResponse:
    """Store the user's Gemini API key encrypted in Supabase."""
    try:
        service.set_user_key(user_id=user_id, plaintext_key=body.key)
        return SuccessResponse(success=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete("/gemini", response_model=SuccessResponse)
async def delete_gemini_key(
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> SuccessResponse:
    """Remove the user's stored Gemini API key."""
    service.delete_user_key(user_id=user_id)
    return SuccessResponse(success=True)
