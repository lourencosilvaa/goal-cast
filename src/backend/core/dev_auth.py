"""Local-only development auth bypass.

Reads an explicit opt-in flag from the environment. This is intended **only**
for local development and must never be enabled in a deployed environment.
Absent (or any value other than the literal ``"true"``) keeps normal Supabase
authentication active.
"""

import os

# Synthetic defaults for the mock local user. Real, environment-specific values
# can override these via ``DEV_USER_ID`` / ``DEV_USER_EMAIL``.
DEFAULT_DEV_USER_ID = "dev-user"
DEFAULT_DEV_USER_EMAIL = "dev@localhost"

_ENABLED_VALUE = "true"


def is_enabled() -> bool:
    """Return True only when ``DEV_AUTH_BYPASS`` is exactly ``"true"``."""
    return os.environ.get("DEV_AUTH_BYPASS", "").strip().lower() == _ENABLED_VALUE


def dev_user_id() -> str:
    """The user_id impersonated while the bypass is active."""
    return os.environ.get("DEV_USER_ID", DEFAULT_DEV_USER_ID)


def dev_user_email() -> str:
    """The email of the mock local user."""
    return os.environ.get("DEV_USER_EMAIL", DEFAULT_DEV_USER_EMAIL)
