from __future__ import annotations

import logging


def get_logger(name: str = "llm_communicator") -> logging.Logger:
    """Return a configured lightweight logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

