"""Tests for the local-only dev auth bypass helper."""


class TestIsEnabled:
    def test_default_is_disabled(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.delenv("DEV_AUTH_BYPASS", raising=False)
        assert dev_auth.is_enabled() is False

    def test_true_enables(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.setenv("DEV_AUTH_BYPASS", "true")
        assert dev_auth.is_enabled() is True

    def test_any_other_value_is_disabled(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.setenv("DEV_AUTH_BYPASS", "1")
        assert dev_auth.is_enabled() is False

        monkeypatch.setenv("DEV_AUTH_BYPASS", "yes")
        assert dev_auth.is_enabled() is False

        monkeypatch.setenv("DEV_AUTH_BYPASS", "false")
        assert dev_auth.is_enabled() is False

    def test_mixed_case_and_whitespace_enables(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.setenv("DEV_AUTH_BYPASS", "  TRUE  ")
        assert dev_auth.is_enabled() is True

    def test_empty_value_is_disabled(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.setenv("DEV_AUTH_BYPASS", "   ")
        assert dev_auth.is_enabled() is False


class TestDevUser:
    def test_default_user_id(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.delenv("DEV_USER_ID", raising=False)
        assert dev_auth.dev_user_id() == dev_auth.DEFAULT_DEV_USER_ID

    def test_custom_user_id(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.setenv("DEV_USER_ID", "my-local-uid")
        assert dev_auth.dev_user_id() == "my-local-uid"

    def test_default_email(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.delenv("DEV_USER_EMAIL", raising=False)
        assert dev_auth.dev_user_email() == dev_auth.DEFAULT_DEV_USER_EMAIL

    def test_custom_email(self, monkeypatch) -> None:
        from src.backend.core import dev_auth

        monkeypatch.setenv("DEV_USER_EMAIL", "me@local")
        assert dev_auth.dev_user_email() == "me@local"
