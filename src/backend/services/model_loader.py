"""Downloads ML models and datasets from a Hugging Face Hub repository."""

from pathlib import Path
from typing import Optional


class ModelLoader:
    def __init__(self, repo_id: str, hf_token: str, local_dir: Path) -> None:
        self.repo_id = repo_id
        self.hf_token = hf_token
        self.local_dir = local_dir
        self._downloaded_path: Optional[Path] = None

    def download(self) -> Path:
        from huggingface_hub import snapshot_download

        try:
            path = snapshot_download(
                repo_id=self.repo_id,
                token=self.hf_token,
                local_dir=self.local_dir,
            )
            self._downloaded_path = Path(path)
            return self._downloaded_path
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download model from Hugging Face repo"
                f" '{self.repo_id}': {exc}"
            ) from exc

    def get_model_path(self, filename: str) -> Path:
        if self._downloaded_path is None:
            raise RuntimeError("Model not downloaded yet — call download() first")
        return self._downloaded_path / filename

    def get_dataset_dir(self, subfolder: str) -> Path:
        if self._downloaded_path is None:
            raise RuntimeError("Model not downloaded yet — call download() first")
        return self._downloaded_path / subfolder
