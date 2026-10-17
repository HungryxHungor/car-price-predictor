# Car Price Prediction API

A FastAPI service serving predictions from the XGBoost model built in `car_price_prediction.ipynb`
(trained on the 100,000 UK Used Car Dataset — 7.01% validation MAPE).

## Files

- `main.py` — the FastAPI app (the API you actually run)
- `train_and_export.py` — retrains the model and saves `model_artifacts.joblib`
  (run this again if you retrain on new data)
- `model_artifacts.joblib` — the trained model + exact preprocessing mappings
  (target-encoded model averages, one-hot column layout, valid category lists)
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000/docs** — this is FastAPI's free interactive Swagger UI.
You can send test requests directly from the browser, no curl needed.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "A1",
    "Make": "audi",
    "year": 2018,
    "transmission": "Manual",
    "mileage": 18500,
    "fuelType": "Petrol",
    "tax": 145.0,
    "mpg": 55.4,
    "engineSize": 1.4
  }'
```

Response:
```json
{"predicted_price": 15215.47, "currency": "GBP", "model_seen_in_training": true, "note": null}
```

## Design notes (worth understanding, not just running)

- **Model loads once at startup**, not per request — check the top of `main.py`.
- **Preprocessing mirrors the notebook exactly** — same target-encoding mapping, same one-hot
  column order — loaded from `model_artifacts.joblib` rather than recomputed, so training and
  serving can never silently drift apart.
- **Unseen `model` values don't crash** — they fall back to the global mean price and the
  response flags `model_seen_in_training: false` so the consumer knows to trust it less.
- **Unknown `Make`/`transmission`/`fuelType` are rejected with a 400** and a list of valid
  values — these are keys into one-hot columns, so silently accepting an unknown one would
  produce a wrong prediction with no warning, which is worse than an explicit error.
- **Malformed types (e.g. mileage as a string) are rejected automatically with a 422** —
  this comes free from the Pydantic schema, no manual validation code needed.

