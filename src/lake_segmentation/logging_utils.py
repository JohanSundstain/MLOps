"""Logging configuration.

A single place that configures the root logger of the project: a stream handler for the
console and, optionally, a file handler inside the ``plots`` directory so that the run logs
are stored together with the figures, as required by the assignment.

``warnings`` are never silenced: filters are not installed on the ``warnings`` module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGER_NAME = "lake_segmentation"

_configured_file: Path | None = None


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger (a child of the ``lake_segmentation`` logger)."""

    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure console logging and, optionally, a log file.

    Repeated calls are harmless: handlers are added only once per log file.
    """

    global _configured_file

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        _configured_file = None

    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    ):
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
        if _configured_file != log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            logger.addHandler(file_handler)
            _configured_file = log_path

    return logger
