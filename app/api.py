from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.pkl"

app = FastAPI(title="Wine Classifier API", version="1.0.0")

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


class WineFeatures(BaseModel):
    alcohol: float = Field(..., example=13.2)
    malic_acid: float = Field(..., example=1.78)
    ash: float = Field(..., example=2.14)
    alcalinity_of_ash: float = Field(..., example=11.2)
    magnesium: float = Field(..., example=100.0)
    total_phenols: float = Field(..., example=2.65)
    flavanoids: float = Field(..., example=2.76)
    nonflavanoid_phenols: float = Field(..., example=0.26)
    proanthocyanins: float = Field(..., example=1.28)
    color_intensity: float = Field(..., example=4.38)
    hue: float = Field(..., example=1.05)
    od280_od315: float = Field(..., example=3.40)
    proline: float = Field(..., example=1050.0)


class PredictionResponse(BaseModel):
    prediction: int
    label: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures) -> PredictionResponse:
    artifact = get_artifact()
    model = artifact["model"]
    target_names = artifact["target_names"]

    row = [[
        features.alcohol,
        features.malic_acid,
        features.ash,
        features.alcalinity_of_ash,
        features.magnesium,
        features.total_phenols,
        features.flavanoids,
        features.nonflavanoid_phenols,
        features.proanthocyanins,
        features.color_intensity,
        features.hue,
        features.od280_od315,
        features.proline,
    ]]
    pred = int(model.predict(row)[0])
    return PredictionResponse(prediction=pred, label=target_names[pred])
