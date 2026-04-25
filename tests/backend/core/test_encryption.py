"""Phase 1 tests for encryption utilities — must FAIL before implementation."""

import pytest
from src.backend.core.encryption import EncryptionService


class TestEncryptionService:
    def setup_method(self) -> None:
        self.key = EncryptionService.generate_key()
        self.service = EncryptionService(fernet_key=self.key)

    def test_generate_key_returns_bytes(self) -> None:
        key = EncryptionService.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0

    def test_encrypt_returns_string(self) -> None:
        token = self.service.encrypt("my-api-key")
        assert isinstance(token, str)
        assert token != "my-api-key"

    def test_decrypt_round_trip(self) -> None:
        plaintext = "AIzaSy_test_gemini_key_123"
        encrypted = self.service.encrypt(plaintext)
        decrypted = self.service.decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_encryptions_of_same_value(self) -> None:
        """Fernet uses random IV — same plaintext yields different ciphertext."""
        a = self.service.encrypt("same-value")
        b = self.service.encrypt("same-value")
        assert a != b

    def test_decrypt_wrong_key_raises(self) -> None:
        other_key = EncryptionService.generate_key()
        other_service = EncryptionService(fernet_key=other_key)
        encrypted = self.service.encrypt("secret")
        with pytest.raises(Exception):
            other_service.decrypt(encrypted)

    def test_decrypt_tampered_token_raises(self) -> None:
        encrypted = self.service.encrypt("secret")
        tampered = encrypted[:-4] + "XXXX"
        with pytest.raises(Exception):
            self.service.decrypt(tampered)

    def test_encrypt_empty_string(self) -> None:
        token = self.service.encrypt("")
        assert self.service.decrypt(token) == ""
