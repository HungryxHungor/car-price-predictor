import math


import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import CarFeatures, PredictionResponse
from .config import ARTIFACT_PATH, REQUIRED_KEYS
from .artifacts import load_artifacts

# 1. Configuration and model loading




artifacts = load_artifacts()

missing_keys = REQUIRED_KEYS - artifacts.keys()
if missing_keys:
    raise RuntimeError(f"Missing artifact keys: {sorted(missing_keys)}")

model = artifacts["model"]
model_mapping = artifacts["model_mapping"]
global_mean_price = artifacts["global_mean_price"]
feature_columns = artifacts["feature_columns"]
onehot_columns = artifacts["onehot_columns"]

valid_makes = artifacts["valid_makes"]
valid_transmissions = artifacts["valid_transmissions"]
valid_fuel_types = artifacts["valid_fuel_types"]



# 2. FastAPI application

app = FastAPI(
    title="Car Price Prediction API",
    description="Predicts used car resale prices using XGBoost.",
    version="1.0.0",
)


# 4. Input normalization

def canonical_category(value: str, allowed_values, field_name: str) -> str:
    """to match user input to the exact category spelling"""

    lookup = {
        str(item).strip().casefold(): item
        for item in allowed_values
    }

    result = lookup.get(value.strip().casefold())

    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown {field_name}: '{value}'. "
                   f"Choose a supported value.",
        )

    return result





# 5. Preprocessing

def preprocess(
    car: CarFeatures,
    make: str,
    transmission: str,
    fuel_type: str,
) -> tuple[pd.DataFrame, bool]:

    # match model names without treating capitalization
    # differences as completely different models.

    model_lookup = {
        str(name).strip().casefold(): name
        for name in model_mapping
    }

    model_key = model_lookup.get(car.model_name.casefold())
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

    # initialize every one-hot feature to zero.
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

    #exact feature order
    X = pd.DataFrame(
        [{col: row[col] for col in feature_columns}],
        columns=feature_columns,
    )

    return X, model_seen





# 6. api Endpoints


@app.get("/health")
def health():
    return {"status": "okay"}


@app.post("/predict", response_model=PredictionResponse)
def predict(car: CarFeatures): #use the pydantic model to validate the input JSON

    make = canonical_category(car.Make, valid_makes, "Make")
    transmission = canonical_category(
        car.transmission, valid_transmissions, "transmission"
    )
    fuel_type = canonical_category(
        car.fuelType, valid_fuel_types, "fuelType"
    )

    try:
        X, model_seen = preprocess(
            car, make, transmission, fuel_type
        )

        price = float(model.predict(X)[0])

        if not math.isfinite(price):
            raise RuntimeError("Model returned a non-finite prediction.")

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server logs.",
        ) from exc

    note = None
    if not model_seen:
        note = (
            f"Model '{car.model_name}' was not seen during training. "
            "The prediction may be less reliable."
        )

    return PredictionResponse(
        predicted_price=round(price, 2),
        model_seen_in_training=model_seen,
        note=note,
    )
