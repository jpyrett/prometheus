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
    """Folder that holds the DataAgent's news files (NEWS_OUTPUT_DIR, the DataAgent's own setting name)."""
    return path_setting("NEWS_OUTPUT_DIR", "news")
