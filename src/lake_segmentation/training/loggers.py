"""Experiment tracking setup.

The course expects MLflow on ``127.0.0.1:8080``. The logger records the hyper-parameters,
the git commit of the run and the figures from the ``plots`` directory. When the tracking
server is unreachable (offline work, fresh clone) the project falls back to a CSV logger so
that a training run never crashes because of the tracking backend.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from lightning.pytorch.loggers import CSVLogger, MLflowLogger
from omegaconf import DictConfig

from lake_segmentation.logging_utils import get_logger
from lake_segmentation.utils import get_git_commit, get_git_dirty

LOGGER = get_logger("training.loggers")


def flatten_config(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config into ``key.subkey`` scalars (MLflow requires flat params)."""

    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, name))
        elif isinstance(value, (list, tuple)):
            if all(not isinstance(item, (Mapping, list, tuple)) for item in value):
                flat[name] = str(list(value))
            else:
                for index, item in enumerate(value):
                    if isinstance(item, Mapping):
                        flat.update(flatten_config(item, f"{name}.{index}"))
                    else:
                        flat[f"{name}.{index}"] = str(item)
        elif isinstance(value, bool) or value is None:
            flat[name] = str(value)
        elif isinstance(value, (int, float, str)):
            flat[name] = value
        else:
            flat[name] = str(value)
    return flat


def is_server_available(tracking_uri: str, timeout: float = 2.0) -> bool:
    """Check that the MLflow tracking server accepts TCP connections."""

    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return Path(tracking_uri).exists() if parsed.scheme == "file" else False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_logger(cfg: DictConfig, save_dir: Path, run_name: str | None = None) -> tuple[Any, bool]:
    """Return ``(logger, is_mlflow)``.

    MLflow is used when the server is reachable, otherwise a CSV logger writes to
    ``logs/lightning/<run_name>``.
    """

    mlflow_cfg = cfg.mlflow
    tracking_uri = str(mlflow_cfg.tracking_uri)
    name = run_name or str(mlflow_cfg.run_name)
    if is_server_available(tracking_uri):
        logger = MLflowLogger(
            experiment_name=str(mlflow_cfg.experiment_name),
            tracking_uri=tracking_uri,
            run_name=name,
            log_model=False,
            save_dir=str(save_dir),
        )
        LOGGER.info(
            "Logging to MLflow at %s (experiment %s)", tracking_uri, mlflow_cfg.experiment_name
        )
        return logger, True

    if not bool(mlflow_cfg.get("allow_fallback", True)):
        raise ConnectionError(
            f"MLflow tracking server is not reachable at {tracking_uri}. Start it with "
            "`mlflow server --host 127.0.0.1 --port 8080` or set mlflow.tracking_uri."
        )
    LOGGER.warning(
        "MLflow is not reachable at %s, falling back to the CSV logger in %s",
        tracking_uri,
        save_dir / "lightning",
    )
    return CSVLogger(save_dir=str(save_dir / "lightning"), name=name), False


def log_run_metadata(logger: Any, cfg: DictConfig) -> None:
    """Log hyper-parameters, the git commit and the full config as an artefact."""

    flat = flatten_config(cfg)
    try:
        logger.log_hyperparams(flat)
        if isinstance(logger, MLflowLogger):
            experiment = logger.experiment
            experiment.set_tags(
                logger.run_id,
                {
                    "git_commit": get_git_commit(),
                    "git_dirty": str(get_git_dirty()),
                    "project": str(cfg.project_name),
                },
            )
            LOGGER.info("Logged run metadata to MLflow run %s", logger.run_id)
    except Exception as error:  # noqa: BLE001 - tracking must not break training
        LOGGER.warning("Could not log hyper-parameters: %s", error)
        return

    try:
        from omegaconf import OmegaConf

        config_dump = Path(logger.save_dir if hasattr(logger, "save_dir") else ".") / "config.yaml"
        config_dump.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, config_dump, resolve=True)
        if isinstance(logger, MLflowLogger):
            logger.experiment.log_artifact(logger.run_id, str(config_dump))
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("Could not log the resolved config: %s", error)


def log_artifacts(logger: Any, paths: list[Path], subdirectory: str = "plots") -> None:
    """Upload files to the active MLflow run (no-op for the CSV logger)."""

    if not isinstance(logger, MLflowLogger):
        return
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        try:
            logger.experiment.log_artifact(logger.run_id, str(path), artifact_path=subdirectory)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("Could not log artefact %s: %s", path, error)
