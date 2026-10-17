"""Configuration constants for the car price API."""
from pathlib import Path

ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "model_artifacts.joblib"
)

REQUIRED_KEYS = {
    "model",
    "model_mapping",
    "global_mean_price",
    "feature_columns",
    "onehot_columns",
    "valid_makes",
    "valid_transmissions",
    "valid_fuel_types",
}