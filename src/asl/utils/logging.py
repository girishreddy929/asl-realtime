"""Project-wide logging setup. Use get_logger(__name__) in modules, never print()."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_LOGGER_NAME = "asl"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int | str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure the 'asl' logger. Safe to call repeatedly (handlers are replaced, not stacked)."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate lines via the root logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(_FORMAT, datefmt="%H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child of the 'asl' logger (names outside the namespace are nested under it)."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")