"""FastAPI dependencies for JWT authentication and authorisation via Supabase."""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.backend.core.supabase_client import get_supabase_client

_bearer = HTTPBearer(auto_error=False)


def _validate_token(token: str) -> str:
    """Verify a raw JWT with Supabase and return the user_id."""
    try:
        client = get_supabase_client()
        response = client.auth.get_user(token)
        if response is None or response.user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )
        return str(response.user.id)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed",
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """Verify JWT and return user_id. Raises 401 for unauthenticated requests."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    return _validate_token(credentials.credentials)


async def get_approved_user(
    user_id: str = Depends(get_current_user),
) -> str:
    """Like get_current_user but also verifies user_profiles.approved = true."""
    from src.backend.services.user_service import UserService

    svc = UserService(supabase=get_supabase_client())
    profile = svc.get_profile(user_id=user_id)
    if profile is None or not profile.get("approved", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account pending approval or access revoked.",
        )
    return user_id
