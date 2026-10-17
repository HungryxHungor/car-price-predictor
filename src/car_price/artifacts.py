"""Model artifact loading and validation."""
import logging
from pathlib import Path

import joblib

from .config import ARTIFACT_PATH, REQUIRED_KEYS

logger = logging.getLogger(__name__)


def load_artifacts(path: Path = ARTIFACT_PATH) -> dict:
    if not path.is_file():
        logger.critical("Model artifact missing at %s", path)
        raise RuntimeError(f"Model artifact not found: {path}")

    logger.info("Loading model artifacts from %s", path)
    try:
        artifacts = joblib.load(path)
    except Exception:
        logger.exception("Failed to deserialize artifact: %s", path)
        raise

    missing = REQUIRED_KEYS - artifacts.keys()
    if missing:
        logger.critical("Artifact missing required keys: %s", missing)
        raise RuntimeError(f"Incomplete artifact, missing: {missing}")

    logger.info("Artifacts loaded. Keys: %s", list(artifacts.keys()))
    return artifacts

