"""Shared utilities for the Smart Product Intelligence project."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ProjectConfig:
    """Central project configuration."""

    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023"
    category: str = "All_Beauty"
    random_seed: int = 42
    max_samples: int | None = None


def get_logger(name: str = "smart_product_intelligence") -> logging.Logger:
    """Return a configured project logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_directories() -> None:
    """Create local runtime directories used by data and model workflows."""

    for directory in (DATA_DIR, ARTIFACTS_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        get_logger(__name__).warning("TensorFlow is not installed; skipped tf seed.")


def placeholder_response(message: str) -> dict[str, str]:
    """Return a consistent placeholder payload for unfinished workflows."""

    return {
        "status": "placeholder",
        "message": message,
    }

