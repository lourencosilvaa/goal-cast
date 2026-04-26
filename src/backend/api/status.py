"""App status endpoints — retraining flag and build version."""

import os
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from src.backend.core.supabase_client import get_supabase_client
from src.backend.services.app_settings_service import AppSettingsService

router = APIRouter(prefix="/api", tags=["status"])

_RETRAIN_KEY = "retraining"


class HealthResponse(BaseModel):
    status: str
    commit: str
    build_time: str


class VersionResponse(BaseModel):
    commit: str
    build_time: str


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Render health check — returns 200 when the service is up."""
    return HealthResponse(
        status="ok",
        commit=os.environ.get("GIT_SHA", "unknown"),
        build_time=os.environ.get("BUILD_TIME", "unknown"),
    )


@router.get("/version", response_model=VersionResponse)
def get_version() -> VersionResponse:
    """Return the git commit SHA baked into the image at build time."""
    return VersionResponse(
        commit=os.environ.get("GIT_SHA", "unknown"),
        build_time=os.environ.get("BUILD_TIME", "unknown"),
    )


def get_app_settings_service() -> AppSettingsService:
    return AppSettingsService(supabase=get_supabase_client())


class RetrainStatusResponse(BaseModel):
    retraining: bool


class SetRetrainStatusRequest(BaseModel):
    retraining: bool


@router.get("/retrain-status", response_model=RetrainStatusResponse)
def get_retrain_status(
    svc: Annotated[AppSettingsService, Depends(get_app_settings_service)],
) -> RetrainStatusResponse:
    value = svc.get(_RETRAIN_KEY)
    return RetrainStatusResponse(retraining=value == "true")


@router.post("/retrain-status", response_model=RetrainStatusResponse)
def set_retrain_status(
    body: SetRetrainStatusRequest,
    svc: Annotated[AppSettingsService, Depends(get_app_settings_service)],
    x_api_key: Optional[str] = Header(default=None),
) -> RetrainStatusResponse:
    configured_key = os.environ.get("RETRAIN_API_KEY", "")
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrain API key not configured.",
        )
    if x_api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    svc.set(_RETRAIN_KEY, "true" if body.retraining else "false")
    return RetrainStatusResponse(retraining=body.retraining)
