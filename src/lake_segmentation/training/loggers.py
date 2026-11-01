"""Experiment tracking setup.

The course expects MLflow on ``127.0.0.1:8080``. The logger records the hyper-parameters,
the git commit of the run and the figures from the ``plots`` directory. When the tracking
server is unreachable (offline work, fresh clone) the project falls back to a CSV logger so
that a training run never crashes because of the tracking backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from lightning.pytorch.loggers import CSVLogger

try:  # Lightning >= 2.6 renamed the class to ``MLFlowLogger`` (capital F)
    from lightning.pytorch.loggers import MLFlowLogger as MLflowLogger
except ImportError:  # pragma: no cover - older Lightning releases
    from lightning.pytorch.loggers import MLflowLogger  # type: ignore[assignment]
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


def is_server_available(tracking_uri: str, timeout: float = 3.0) -> bool:
    """Whether ``tracking_uri`` answers like an MLflow tracking server.

    A plain TCP check is not enough: in Google Colab, for example, port 8080 is already taken
    by the Jupyter server, which happily accepts connections and then answers every MLflow API
    call with an HTML 404 page. The ``/health`` endpoint is probed instead and HTML answers are
    rejected.
    """

    parsed = urlparse(tracking_uri)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        health_url = f"{tracking_uri.rstrip('/')}/health"
        try:
            with urlopen(health_url, timeout=timeout) as response:  # noqa: S310 - user config
                body = response.read(256).decode("utf-8", errors="ignore")
                if response.status != 200:
                    return False
                return "<html" not in body.lower()
        except (URLError, OSError, ValueError):
            return False

    local_path = (
        tracking_uri[len("file://") :] if tracking_uri.startswith("file://") else tracking_uri
    )
    return Path(local_path).exists()


def _mlflow_logger_reachable(logger: MLflowLogger, timeout_note: str = "") -> bool:
    """Exercise one MLflow API call so that a broken backend is detected before training."""

    try:
        logger.experiment.search_experiments(max_results=1)
        return True
    except Exception as error:  # noqa: BLE001 - any MLflow/HTTP failure means "unusable"
        LOGGER.warning("MLflow backend is not usable (%s)%s", error, timeout_note)
        return False


def build_logger(cfg: DictConfig, save_dir: Path, run_name: str | None = None) -> tuple[Any, bool]:
    """Return ``(logger, is_mlflow)``.

    MLflow is used when the server is reachable, otherwise a CSV logger writes to
    ``logs/lightning/<run_name>``.
    """

    mlflow_cfg = cfg.mlflow
    tracking_uri = str(mlflow_cfg.tracking_uri)
    name = run_name or str(mlflow_cfg.run_name)
    fallback_reason = ""
    if is_server_available(tracking_uri):
        try:
            logger = MLflowLogger(
                experiment_name=str(mlflow_cfg.experiment_name),
                tracking_uri=tracking_uri,
                run_name=name,
                log_model=False,
                save_dir=str(save_dir),
            )
            if _mlflow_logger_reachable(logger):
                LOGGER.info(
                    "Logging to MLflow at %s (experiment %s)",
                    tracking_uri,
                    mlflow_cfg.experiment_name,
                )
                return logger, True
            fallback_reason = "the MLflow API is not reachable"
        except Exception as error:  # noqa: BLE001 - never let tracking break training
            fallback_reason = f"{type(error).__name__}: {error}"
            LOGGER.warning("Could not create the MLflow logger: %s", error)
    else:
        fallback_reason = "no MLflow server answered on /health"

    if not bool(mlflow_cfg.get("allow_fallback", True)):
        raise ConnectionError(
            f"MLflow tracking server is not usable at {tracking_uri} ({fallback_reason}). "
            "Start it with `mlflow server --host 127.0.0.1 --port 8080` or set "
            "mlflow.tracking_uri. In Google Colab port 8080 is taken by Jupyter, use "
            "another port (see the training notebook)."
        )
    LOGGER.warning(
        "Falling back to the CSV logger in %s (%s)", save_dir / "lightning", fallback_reason
    )
    return CSVLogger(save_dir=str(save_dir / "lightning"), name=name), False


def log_run_metadata(logger: Any, cfg: DictConfig) -> None:
    """Log hyper-parameters, the git commit and the full config as an artefact."""

    flat = flatten_config(cfg)
    try:
        logger.log_hyperparams(flat)
        if isinstance(logger, MLflowLogger):
            experiment = logger.experiment
            for key, value in {
                "git_commit": get_git_commit(),
                "git_dirty": str(get_git_dirty()),
                "project": str(cfg.project_name),
            }.items():
                experiment.set_tag(logger.run_id, key, value)
            LOGGER.info("Logged run metadata to MLflow run %s", logger.run_id)
    except Exception as error:  # noqa: BLE001 - tracking must not break training
        LOGGER.warning("Could not log hyper-parameters: %s", error)
        return

    try:
        from omegaconf import OmegaConf

        # MLflowLogger has ``save_dir=None``, CSVLogger has a real directory.
        base_dir = getattr(logger, "save_dir", None) or "."
        config_dump = Path(base_dir) / "config.yaml"
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
