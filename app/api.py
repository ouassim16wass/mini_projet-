from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

app = FastAPI(title="Iris Classifier API", version="1.0.0")

_artifact: dict | None = None


def get_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run train.py first.",
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, example=5.1)
    sepal_width: float = Field(..., ge=0, example=3.5)
    petal_length: float = Field(..., ge=0, example=1.4)
    petal_width: float = Field(..., ge=0, example=0.2)


class PredictionResponse(BaseModel):
    prediction: int
    label: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
    artifact = get_artifact()
    model = artifact["model"]
    target_names = artifact["target_names"]

    row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    pred = int(model.predict(row)[0])
    return PredictionResponse(prediction=pred, label=target_names[pred])
