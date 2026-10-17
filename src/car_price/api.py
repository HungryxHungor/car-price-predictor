import math

import logging

from fastapi import FastAPI, HTTPException

from .schemas import CarFeatures, PredictionResponse
from .artifacts import load_artifacts
from .preprocessing import canonical_category, preprocess


# Configuration and model loading

logger = logging.getLogger(__name__)

artifacts = load_artifacts()


model = artifacts["model"]

valid_makes = artifacts["valid_makes"]
valid_transmissions = artifacts["valid_transmissions"]
valid_fuel_types = artifacts["valid_fuel_types"]



# FastAPI application

app = FastAPI(
    title="Car Price Prediction API",
    description="Predicts used car resale prices using XGBoost.",
    version="1.0.0",
)





# api Endpoints

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
            car, make, transmission, fuel_type, artifacts
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
