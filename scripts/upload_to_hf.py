"""
Upload trained model artefacts and datasets to a Hugging Face Hub private repository.

Model artefacts (.joblib, .json) are uploaded to the repo root.
Dataset CSVs are converted to per-league Parquet files and uploaded under datasets/.

Usage:
    uv run python scripts/upload_to_hf.py
    uv run python scripts/upload_to_hf.py --models-dir output/models
    uv run python scripts/upload_to_hf.py --repo-id myuser/my-model-repo

Environment variables (override CLI args):
    HF_TOKEN      — HuggingFace token with write access to the repo
    HF_REPO_ID    — e.g. myuser/football-prediction-model
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    import pandas as pd

load_dotenv()

from config.config_loader import load_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload model artefacts and datasets to HF Hub"
    )
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("MODELS_DIR", "output/models"),
        help="Local directory containing model artefacts",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("CACHE_DIR", "datasets/cache"),
        help="Local directory containing raw CSV cache files",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID", ""),
        help="HuggingFace repo ID, e.g. myuser/football-model",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help=(
            "Publish only the Parquet datasets, leaving model artefacts alone. "
            "Used when new data arrived but the refit was held back."
        ),
    )
    return parser.parse_args()


#: Name the Space looks for when loading cross-league history.
EUROPEAN_DATASET = "european.parquet"

#: Columns the loader adds itself and therefore must survive the trim.
#: ``Season`` is what ``_load_from_hf_parquet`` filters on; ``League`` is
#: filled in on read when absent, so it need not be uploaded.
EXTRA_DATASET_COLUMNS = ("Season", "League")


def _keep_columns(frame: "pd.DataFrame", config: object) -> "pd.DataFrame":
    """Drop columns the loader discards on read.

    Both readers apply ``columns_to_keep`` immediately after ``read_parquet``,
    so anything outside it is dead weight — and the *cause* of the upload
    failing: old football-data extras like ``BbAH`` read as int64 in one
    season and object in another, and concatenating seasons yields a column
    pyarrow refuses to write. Trimming is lossless by construction.
    """
    wanted = list(config.data.columns_to_keep) + list(  # type: ignore[attr-defined]
        EXTRA_DATASET_COLUMNS
    )
    available = [column for column in wanted if column in frame.columns]
    return frame[available]


def _normalise_dtypes(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Give every object column a single, writable type.

    A belt-and-braces guard for the columns that *are* kept: if a future
    season introduces a stray value in, say, ``HS``, the concatenated column
    would again hold two types. Numeric where every value converts, string
    otherwise — deterministic either way, and never a mix.
    """
    import pandas as pd

    result = frame.copy()
    for column in result.columns:
        if result[column].dtype != object:
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        if numeric.notna().sum() == result[column].notna().sum():
            result[column] = numeric
        else:
            result[column] = result[column].astype("string")
    return result


def _build_parquet_datasets(cache_dir: Path, tmp_dir: Path) -> dict[str, Path]:
    """Convert per-season CSVs to per-league Parquet files.

    Returns mapping of league_code → parquet_path.
    """
    import pandas as pd

    csv_files = list(cache_dir.glob("????_??.csv")) + list(
        cache_dir.glob("????_???.csv")
    )
    if not csv_files:
        print(f"WARNING: no season CSV files found in {cache_dir}")
        return {}

    # Group CSVs by league code (filename format: {season}_{league}.csv)
    by_league: dict[str, list[Path]] = {}
    for csv_path in sorted(csv_files):
        parts = csv_path.stem.split("_", 1)
        if len(parts) != 2:
            continue
        season, league = parts
        by_league.setdefault(league, []).append((season, csv_path))

    datasets_dir = tmp_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    for league, season_files in by_league.items():
        frames = []
        for season, csv_path in season_files:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
                df["Season"] = season
                frames.append(df)
            except Exception as e:
                print(f"  WARNING: could not read {csv_path.name}: {e}")

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined = _normalise_dtypes(_keep_columns(combined, load_config()))
        parquet_path = datasets_dir / f"{league}.parquet"
        combined.to_parquet(parquet_path, index=False)
        total_rows = len(combined)
        size_kb = parquet_path.stat().st_size / 1024
        print(
            f"  {league}.parquet  {len(season_files)} seasons  {total_rows} rows  {size_kb:.1f} KB"
        )
        result[league] = parquet_path

    return result


def _build_european_dataset(config: object, datasets_dir: Path) -> Path | None:
    """Write the translated European corpus for the Space to read.

    Uploaded already translated to canonical keys, so the Space never needs
    Supabase or the alias registry — it just reads the same rows training
    used, which is what keeps its ELO features identical to the model's.
    """
    from src.corpus.european_corpus import load_european_corpus

    corpus = load_european_corpus(config, verbose=False)
    if corpus.empty:
        print("  no European corpus to upload (ratings will not be cross-league)")
        return None
    path = datasets_dir / EUROPEAN_DATASET
    corpus.to_parquet(path, index=False)
    size_kb = path.stat().st_size / 1024
    print(f"  {EUROPEAN_DATASET}  {len(corpus)} matches  {size_kb:.1f} KB")
    return path


def main(api: object | None = None) -> None:
    args = _parse_args()

    models_dir = Path(args.models_dir)
    artefacts: list[Path] = []

    # In --datasets-only mode the model deliberately was not rebuilt, so an
    # absent or empty models directory is the expected state, not an error.
    if not args.datasets_only:
        if not models_dir.exists():
            print(f"ERROR: models directory not found: {models_dir}")
            sys.exit(1)

        artefacts = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.json"))
        if not artefacts:
            print(f"ERROR: no .joblib or .json artefacts found in {models_dir}")
            sys.exit(1)

        print(f"\nModel artefacts to upload ({len(artefacts)}):")
        for f in sorted(artefacts):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:40s} {size_kb:8.1f} KB")
    else:
        print("\nDatasets-only run — model artefacts left untouched.")

    if not args.repo_id:
        print("\nERROR: --repo-id or HF_REPO_ID env var is required")
        sys.exit(1)

    if not args.token:
        print("\nERROR: --token or HF_TOKEN env var is required")
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    print(f"\nBuilding per-league Parquet datasets from {cache_dir} …")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet_files = _build_parquet_datasets(cache_dir, tmp_path)
        # Cross-league history for the Space, already translated to canonical
        # keys so it needs neither Supabase nor the alias registry.
        european = _build_european_dataset(load_config(), tmp_path / "datasets")
        dataset_count = len(parquet_files) + (1 if european else 0)

        if args.dry_run:
            print(
                f"\n[dry-run] Would upload {len(artefacts)} model artefacts to: {args.repo_id}"
            )
            print(
                f"[dry-run] Would upload {dataset_count} Parquet dataset files to: {args.repo_id}/datasets/"
            )
            return

        if api is None:
            from huggingface_hub import HfApi

            api = HfApi(token=args.token)

        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=True,
            exist_ok=True,
        )

        commit_message = (
            f"Retrain {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        if not args.datasets_only:
            print(f"\nUploading model artefacts to {args.repo_id} …")
            api.upload_folder(
                folder_path=str(models_dir),
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
            )

        if parquet_files:
            datasets_dir = tmp_path / "datasets"
            print(
                f"\nUploading {len(parquet_files)} Parquet dataset files to {args.repo_id}/datasets/ …"
            )
            api.upload_folder(
                folder_path=str(datasets_dir),
                repo_id=args.repo_id,
                path_in_repo="datasets",
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
            )

    print(f"\n✓ Uploaded successfully → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
