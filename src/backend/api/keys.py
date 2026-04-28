"""API routes for managing user API keys (Gemini, NVIDIA)."""

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


class KeyStatus(BaseModel):
    has_key: bool


GeminiKeyStatus = KeyStatus


class KeyRequest(BaseModel):
    key: str


GeminiKeyRequest = KeyRequest


class SuccessResponse(BaseModel):
    success: bool


@router.get("/gemini", response_model=KeyStatus)
async def get_gemini_key_status(
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> KeyStatus:
    key = service.get_user_key(user_id=user_id, service="gemini")
    return KeyStatus(has_key=key is not None)


@router.put("/gemini", response_model=SuccessResponse)
async def save_gemini_key(
    body: KeyRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> SuccessResponse:
    try:
        service.set_user_key(user_id=user_id, plaintext_key=body.key, service="gemini")
        return SuccessResponse(success=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete("/gemini", response_model=SuccessResponse)
async def delete_gemini_key(
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> SuccessResponse:
    service.delete_user_key(user_id=user_id, service="gemini")
    return SuccessResponse(success=True)


@router.get("/nvidia", response_model=KeyStatus)
async def get_nvidia_key_status(
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> KeyStatus:
    key = service.get_user_key(user_id=user_id, service="nvidia")
    return KeyStatus(has_key=key is not None)


@router.put("/nvidia", response_model=SuccessResponse)
async def save_nvidia_key(
    body: KeyRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> SuccessResponse:
    try:
        service.set_user_key(user_id=user_id, plaintext_key=body.key, service="nvidia")
        return SuccessResponse(success=True)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.delete("/nvidia", response_model=SuccessResponse)
async def delete_nvidia_key(
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> SuccessResponse:
    service.delete_user_key(user_id=user_id, service="nvidia")
    return SuccessResponse(success=True)
