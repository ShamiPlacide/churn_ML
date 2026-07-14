from __future__ import annotations

import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import ChurnRequest, ChurnResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python -m src.train` first."
        )
    _model["pipeline"] = joblib.load(MODEL_PATH)
    yield
    _model.clear()


app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts whether a telecom customer is likely to churn.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in _model}


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    if "pipeline" not in _model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([request.model_dump()])
    pipeline = _model["pipeline"]

    proba = float(pipeline.predict_proba(row)[0, 1])
    prediction = "Yes" if proba >= 0.5 else "No"

    return ChurnResponse(
        churn_prediction=prediction,
        churn_probability=round(proba, 4),
        model_version=MODEL_VERSION,
    )
