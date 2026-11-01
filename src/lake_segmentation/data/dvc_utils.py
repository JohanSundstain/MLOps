"""Integration with DVC.

The processed dataset is versioned with DVC instead of git. Training and inference call
:func:`ensure_processed_data`, which pulls the dataset from the configured DVC remote with the
DVC Python API when it is not available locally yet. If the remote is unreachable (a fresh
clone without access to the storage) the caller can fall back to rebuilding the dataset from
the public Sentinel-2 / WorldCover sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig

from lake_segmentation.config import resolve_path
from lake_segmentation.constants import MANIFEST_FILENAME
from lake_segmentation.logging_utils import get_logger

LOGGER = get_logger("data.dvc")


def _open_repo(project_root: Path):
    from dvc.repo import Repo

    return Repo(str(project_root))


def pull_data(
    project_root: Path, targets: Sequence[str] | None = None, remote: str | None = None
) -> bool:
    """``dvc pull`` through the Python API. Returns ``True`` on success."""

    try:
        repo = _open_repo(project_root)
    except Exception as error:  # noqa: BLE001 - dvc raises its own exception hierarchy
        LOGGER.warning("Could not open the DVC repository in %s: %s", project_root, error)
        return False

    try:
        result = repo.pull(targets=list(targets) if targets else None, remote=remote)
    except Exception as error:  # noqa: BLE001 - no remote, no credentials, missing cache, ...
        LOGGER.warning("dvc pull failed: %s", error)
        return False

    fetched = sum(result.get("fetched", {}).values()) if isinstance(result, dict) else 0
    added = sum(result.get("added", {}).values()) if isinstance(result, dict) else 0
    LOGGER.info("dvc pull finished: %d files added, %d files fetched", added, fetched)
    return True


def push_data(
    project_root: Path, targets: Sequence[str] | None = None, remote: str | None = None
) -> bool:
    """``dvc push`` through the Python API. Returns ``True`` on success."""

    try:
        repo = _open_repo(project_root)
        repo.push(targets=list(targets) if targets else None, remote=remote)
    except Exception as error:  # noqa: BLE001 - dvc raises its own exception hierarchy
        LOGGER.warning("dvc push failed: %s", error)
        return False
    LOGGER.info("dvc push finished")
    return True


def ensure_processed_data(cfg: DictConfig, project_root: Path) -> Path:
    """Return the processed data directory, pulling it with DVC if necessary."""

    processed_dir = resolve_path(cfg.paths.data_processed, project_root)
    manifest_path = processed_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        LOGGER.info("Processed dataset found at %s", processed_dir)
        return processed_dir

    dvc_cfg = cfg.data.get("dvc", {})
    if bool(dvc_cfg.get("enabled", True)):
        targets = [str(cfg.paths.data_processed)]
        LOGGER.info("Dataset is missing locally, trying dvc pull for %s", targets)
        remote = dvc_cfg.get("remote") or None
        pull_data(project_root, targets=targets, remote=remote)
        if manifest_path.is_file():
            return processed_dir

    raise FileNotFoundError(
        f"Processed dataset not found at {processed_dir}. Run "
        "`python commands.py preprocess` (or `dvc pull`) before training/inference."
    )
