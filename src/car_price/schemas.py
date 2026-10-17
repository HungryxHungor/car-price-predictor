"""Pydantic request and response schemas for the car price API."""
from pydantic import BaseModel, ConfigDict, Field


class CarFeatures(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "model": "A1",
                "Make": "audi",
                "year": 2018,
                "transmission": "Manual",
                "mileage": 18500,
                "fuelType": "Petrol",
                "tax": 145.0,
                "mpg": 55.4,
                "engineSize": 1.4,
            }
        },
    )

    model_name: str = Field(alias="model", min_length=1, description="Car model")
    Make: str = Field(min_length=1, description="Car manufacturer")
    year: int = Field(ge=1990, le=2026)
    transmission: str = Field(min_length=1)
    mileage: int = Field(ge=0, le=300000)
    fuelType: str = Field(min_length=1)
    tax: float = Field(ge=0, allow_inf_nan=False)
    mpg: float = Field(gt=0, allow_inf_nan=False)
    engineSize: float = Field(gt=0, le=8.0, allow_inf_nan=False)


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "GBP"
    model_seen_in_training: bool
    note: str | None = None