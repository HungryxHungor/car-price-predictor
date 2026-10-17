"""Feature preprocessing for the car price model."""
import logging

import pandas as pd
from fastapi import HTTPException

from .schemas import CarFeatures

logger = logging.getLogger(__name__)


def canonical_category(value: str, allowed_values, field_name: str) -> str:
    """Match user input to the exact category spelling (case-insensitive)."""
    
    lookup = {str(item).strip().casefold(): item for item in allowed_values}
    result = lookup.get(value.strip().casefold())
    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown {field_name}: '{value}'. Choose a supported value.",
        )
    return result


def preprocess(
    car: CarFeatures,
    make: str,
    transmission: str,
    fuel_type: str,
    artifacts: dict,
) -> tuple[pd.DataFrame, bool]:

    model_mapping = artifacts["model_mapping"]
    global_mean_price = artifacts["global_mean_price"]
    onehot_columns = artifacts["onehot_columns"]
    feature_columns = artifacts["feature_columns"]

    model_lookup = {str(name).strip().casefold(): name for name in model_mapping}
    model_key = model_lookup.get(car.model_name.strip().casefold())
    model_seen = model_key is not None

    if model_seen:
        model_encoded = model_mapping[model_key]
    else:
        model_encoded = global_mean_price

    row = {
        "year": car.year,
        "mileage": car.mileage,
        "tax": car.tax,
        "mpg": car.mpg,
        "engineSize": car.engineSize,
        "model_encoded": model_encoded,
    }
    for col in onehot_columns:
        row[col] = 0

    selected_columns = [
        f"Make_{make}",
        f"transmission_{transmission}",
        f"fuelType_{fuel_type}",
    ]
    for col in selected_columns:
        if col not in onehot_columns:
            logger.error("Expected feature missing from artifacts: %s", col)
            raise RuntimeError(
                f"Model artifacts do not contain expected feature: {col}"
            )
        row[col] = 1

    missing_features = set(feature_columns) - row.keys()
    if missing_features:
        logger.error("Missing model features: %s", missing_features)
        raise RuntimeError("Preprocessing did not create all model features.")

    X = pd.DataFrame(
        [{col: row[col] for col in feature_columns}],
        columns=feature_columns,
    )
    return X, model_seen