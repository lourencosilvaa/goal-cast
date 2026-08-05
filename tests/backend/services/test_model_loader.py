"""Phase 1 tests for HuggingFace ModelLoader — must FAIL before implementation."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestModelLoader:
    def _make_loader(self, repo_id: str = "user/football-model", token: str = "hf_test") -> object:
        from src.backend.services.model_loader import ModelLoader
        return ModelLoader(repo_id=repo_id, hf_token=token, local_dir=Path("/tmp/test_hf"))

    def test_loader_stores_config(self) -> None:
        loader = self._make_loader(repo_id="user/repo", token="hf_abc")
        assert loader.repo_id == "user/repo"
        assert loader.hf_token == "hf_abc"

    def test_download_calls_snapshot_download(self) -> None:
        loader = self._make_loader()
        with patch("huggingface_hub.snapshot_download") as mock_dl:
            mock_dl.return_value = "/tmp/test_hf"
            result = loader.download()
            mock_dl.assert_called_once_with(
                repo_id="user/football-model",
                token="hf_test",
                local_dir=Path("/tmp/test_hf"),
            )
            assert result == Path("/tmp/test_hf")

    def test_download_raises_on_failure(self) -> None:
        loader = self._make_loader()
        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("401 Unauthorized"),
        ):
            with pytest.raises(RuntimeError, match="model"):
                loader.download()

    def test_get_model_path_returns_joblib(self) -> None:
        loader = self._make_loader()
        with patch("huggingface_hub.snapshot_download", return_value="/tmp/test_hf"):
            loader.download()
        path = loader.get_model_path(filename="ensemble_model.joblib")
        assert path == Path("/tmp/test_hf") / "ensemble_model.joblib"

    def test_get_model_path_before_download_raises(self) -> None:
        loader = self._make_loader()
        with pytest.raises(RuntimeError, match="download"):
            loader.get_model_path(filename="ensemble_model.joblib")
