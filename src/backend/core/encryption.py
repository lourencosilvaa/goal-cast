"""Fernet symmetric encryption for sensitive values at rest."""

import os

from cryptography.fernet import Fernet, InvalidToken


class EncryptionService:
    def __init__(self, fernet_key: bytes) -> None:
        self._fernet = Fernet(fernet_key)

    @staticmethod
    def generate_key() -> bytes:
        return Fernet.generate_key()

    @staticmethod
    def key_from_env(env_var: str = "ENCRYPTION_KEY") -> bytes:
        raw = os.environ.get(env_var, "")
        if not raw:
            raise RuntimeError(f"Environment variable {env_var!r} is not set")
        return raw.encode()

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except (InvalidToken, Exception) as exc:
            raise ValueError("Decryption failed — invalid token or wrong key") from exc
