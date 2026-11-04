"""Paths and settings for the RAG library.

A setting comes from the environment first, then from `.env` at the repo root, then from a default.
These are functions, not constants, so nothing is read at import time and a test can override a
setting with an environment variable.
"""
import os
from pathlib import Path

from dotenv import dotenv_values

# This file is prometheus/rag/config.py, so the repo root is two folders up.
REPO_ROOT = Path(__file__).resolve().parents[2]


def setting(name: str, default: str) -> str:
    """One setting: environment variable, then .env at the repo root, then the default."""
    return os.environ.get(name) or dotenv_values(REPO_ROOT / ".env").get(name) or default


def path_setting(name: str, default: str) -> Path:
    """A path setting. A relative path is resolved against the repo root."""
    path = Path(setting(name, default)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def news_dir() -> Path:
    """Folder that holds the DataToolbox's news files (NEWS_OUTPUT_DIR, the DataToolbox's own setting name)."""
    return path_setting("NEWS_OUTPUT_DIR", "news")


def press_releases_dir() -> Path:
    """Folder that holds the DataToolbox's press release files, one per symbol: <SYMBOL>_PRs.txt (PRESS_RELEASES_OUTPUT_DIR).
    The default is the folder the toolbox has always written to, so no file moves (design choices entry 34)."""
    return path_setting("PRESS_RELEASES_OUTPUT_DIR", "data/press_releases")


def filings_dir() -> Path:
    """Folder that holds the DataToolbox's SEC filing files, one per symbol: <SYMBOL>.txt (FILINGS_OUTPUT_DIR).
    The default is the folder the toolbox has always written to, so no file moves (design choices entry 35)."""
    return path_setting("FILINGS_OUTPUT_DIR", "data/filings")


def vector_store_path() -> Path:
    """Folder where Chroma keeps its files. It is derived data: delete it and ingest again to rebuild."""
    return path_setting("VECTOR_STORE_PATH", "data/processed/vectorstore")


def embedding_model() -> str:
    """The sentence-transformers model that turns text into vectors."""
    return setting("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def embedding_device() -> str:
    """Where the embedding model runs: cpu (default) or cuda. The scores are the same either way."""
    return setting("EMBEDDING_DEVICE", "cpu")
