"""Hydra configuration loading and project path resolution.

The project never relies on the current working directory: every relative path from a
config is resolved against the *project root*, i.e. the directory that contains
``pyproject.toml`` and the ``configs`` folder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from lake_segmentation.constants import CONFIGS_DIRNAME

PROJECT_ROOT_ENV_VAR = "LAKE_SEGMENTATION_ROOT"
CONFIG_DIR_ENV_VAR = "LAKE_SEGMENTATION_CONFIGS"


def find_project_root(start: Path | None = None) -> Path:
    env_root = os.environ.get(PROJECT_ROOT_ENV_VAR)
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd().resolve())
    candidates.append(Path(__file__).resolve())

    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "pyproject.toml").is_file() and (parent / CONFIGS_DIRNAME).is_dir():
                return parent
    return Path.cwd().resolve()


def find_config_dir(config_dir: str | Path | None = None) -> Path:
    if config_dir is not None:
        path = Path(config_dir).expanduser().resolve()
    elif os.environ.get(CONFIG_DIR_ENV_VAR):
        path = Path(os.environ[CONFIG_DIR_ENV_VAR]).expanduser().resolve()
    else:
        path = find_project_root() / CONFIGS_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(f"Config directory does not exist: {path}")
    return path


def parse_overrides(overrides: str | Sequence[str] | None) -> list[str]:
    if overrides is None:
        return []
    if isinstance(overrides, str):
        items: Iterable[str] = overrides.replace(",", " ").split()
    else:
        items = overrides
    result: list[str] = []
    for item in items:
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result


def load_config(
    config_name: str = "config",
    overrides: str | Sequence[str] | None = None,
    config_dir: str | Path | None = None,
) -> DictConfig:
    resolved_config_dir = find_config_dir(config_dir)
    hydra_overrides = parse_overrides(overrides)
    with initialize_config_dir(config_dir=str(resolved_config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    return cfg


def to_container(cfg: DictConfig | dict) -> dict:
    """Return a plain Python container for a config (used for serialisation)."""

    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[return-value]
    return dict(cfg)


def resolve_path(path_value: str | Path, root: Path | None = None) -> Path:
    """Resolve a possibly relative path from the config against the project root."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    project_root = root if root is not None else find_project_root()
    return (project_root / path).resolve()


def resolve_paths(cfg: DictConfig) -> dict[str, Path]:
    root = find_project_root()
    resolved: dict[str, Path] = {}
    for key, value in cfg.paths.items():
        path = resolve_path(value, root)
        path.mkdir(parents=True, exist_ok=True)
        resolved[key] = path
    return resolved
