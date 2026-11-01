"""Small shared helpers: reproducibility, git metadata and timing."""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from lake_segmentation.config import find_project_root


def seed_everything(seed: int) -> None:
    """Make a run reproducible (PyTorch is imported lazily so that light CLIs stay fast)."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the project
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_seed(*parts: object) -> int:
    """Deterministic 64-bit seed from arbitrary values.

    Python's built-in ``hash()`` is salted per process for strings, so it must never be used
    where reproducibility matters (for example when seeding the per-scene patch sampling).
    """

    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def get_git_commit(short: bool = False) -> str:
    """Return the current git commit hash or ``"unknown"`` outside of a repository."""

    arguments = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            arguments,
            cwd=find_project_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def get_git_dirty() -> bool:
    """Whether the working tree contains uncommitted changes."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=find_project_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(result.stdout.strip())


@contextmanager
def timer(label: str = "operation") -> Iterator[dict[str, float]]:
    """Measure the duration of a block of code."""

    state: dict[str, float] = {}
    started = time.perf_counter()
    try:
        yield state
    finally:
        state["seconds"] = time.perf_counter() - started
        state["label"] = label  # type: ignore[assignment]


def human_readable_bytes(num_bytes: float) -> str:
    """Format a byte count for log messages."""

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PiB"


def directory_size_bytes(path: Path) -> int:
    """Total size of all files under ``path``."""

    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
