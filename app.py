"""FastAPI application for claim outcome predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator

from train import MODEL_FILE

pipeline = None
loaded_model_mtime_ns: int | None = None


def load_model() -> None:
    """Load the artifact or raise an explicit setup error."""
    global pipeline, loaded_model_mtime_ns
    model_path = Path(MODEL_FILE)
    if not model_path.is_file():
        raise RuntimeError(f"Model file '{MODEL_FILE}' not found. Run 'python train.py' first.")
    loaded_pipeline = joblib.load(model_path)
    pipeline = loaded_pipeline
    loaded_model_mtime_ns = model_path.stat().st_mtime_ns


def refresh_model_if_changed() -> None:
    """Reload a model artifact after training replaces it, without restarting the API."""
    model_path = Path(MODEL_FILE)
    if not model_path.is_file():
        raise RuntimeError(f"Model file '{MODEL_FILE}' not found. Run 'python train.py' first.")
    if pipeline is None or model_path.stat().st_mtime_ns != loaded_model_mtime_ns:
        load_model()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Claim Outcome Prediction API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    details = [
        f"Field '{' -> '.join(str(part) for part in error['loc'])}': {error['msg']}"
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid request", "details": details},
    )


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payer: StrictStr
    provider: StrictStr
    cpt_code: StrictStr
    diagnosis_code: StrictStr
    billed_amount: StrictFloat = Field(gt=0, allow_inf_nan=False)
    patient_age: StrictInt = Field(ge=0, le=125)

    @field_validator("payer", "provider", "cpt_code", "diagnosis_code")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


@app.get("/health")
def health() -> dict[str, Any]:
    refresh_model_if_changed()
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": list(pipeline.classes_),
    }


@app.post("/predict")
def predict(claim: ClaimRequest) -> dict[str, Any]:
    refresh_model_if_changed()
    features = pd.DataFrame([claim.model_dump()])
    probabilities = pipeline.predict_proba(features)[0]
    return {
        "predicted_status": str(pipeline.classes_[probabilities.argmax()]),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(pipeline.classes_, probabilities)
        },
    }

