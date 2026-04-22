from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    def doRollover(self) -> None:
        try:
            super().doRollover()
        except PermissionError:
            # Windows may temporarily lock the active log file when multiple
            # threads/processes touch the same handler around midnight. Keep
            # writing to the current file instead of crashing the caller.
            return


def get_logger(name: str, log_dir: str | Path = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = SafeTimedRotatingFileHandler(
        filename=Path(log_dir) / f"{name}.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
