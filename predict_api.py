"""
predict_api.py
--------------
FastAPI microservice for cinema demand prediction using XGBoost.

Endpoint:
  POST /predict-demand
  Body : { show_hour, day_of_week, seat_occupancy_pct, movie_popularity, recent_bookings }
  Response: { demand_score: float (0-1) }

Run:
  uvicorn predict_api:app --reload --port 8085
"""

import os
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "demand_model.pkl")

# ── Load model at startup ──────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Please run `python train_model.py` first."
    )

model = joblib.load(MODEL_PATH)
print(f"[*] XGBoost model loaded from {MODEL_PATH}")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cinema Demand Prediction API",
    description="Predicts show demand (0–1) using XGBoost. Used for dynamic discount pricing.",
    version="1.0.0",
)

# Allow requests from the Node.js backend (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:3000", "*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Request / Response schemas ─────────────────────────────────────────────────
class DemandRequest(BaseModel):
    show_hour: float = Field(..., description="Hour of day")
    day_of_week: float = Field(..., description="Day of week")
    seat_occupancy_pct: float = Field(..., description="Occupancy fraction or pct")
    movie_popularity: float = Field(..., description="Movie Popularity")
    minutes_until_show: float = Field(60.0, description="Minutes until show")
    recent_bookings: float = Field(0.0, description="Bookings in last 60m")

    class Config:
        json_schema_extra = {
            "example": {
                "show_hour": 22,
                "day_of_week": 1,
                "seat_occupancy_pct": 0.10,
                "movie_popularity": 0.20,
                "recent_bookings": 3,
            }
        }

class DemandResponse(BaseModel):
    demand_score: float = Field(..., description="Predicted demand score between 0 and 1")
    demand_level: str   = Field(..., description="LOW / MEDIUM / HIGH label")

# ── Helper ─────────────────────────────────────────────────────────────────────
def demand_label(score: float) -> str:
    if score < 0.4:
        return "LOW"
    elif score < 0.7:
        return "MEDIUM"
    return "HIGH"

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Cinema Demand Prediction API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.post("/predict-demand", response_model=DemandResponse, tags=["Prediction"])
def predict_demand(payload: DemandRequest):
    """
    Predict demand score (0–1) for a movie show.

    - **show_hour**: Hour when the show starts (0-23)
    - **day_of_week**: 0=Monday … 6=Sunday
    - **seat_occupancy_pct**: Fraction of seats already booked (0.0–1.0)
    - **movie_popularity**: Normalised popularity/rating score (0.0–1.0)
    - **recent_bookings**: Bookings made in the last 60 minutes
    """
    try:
        features = [[
            float(payload.show_hour),
            float(payload.day_of_week),
            float(payload.seat_occupancy_pct),
            float(payload.movie_popularity),
            float(payload.recent_bookings),
        ]]
        if hasattr(model, 'predict_proba'):
            raw_score = float(model.predict_proba(features)[0][1])
        else:
            raw_score = float(model.predict(features)[0])
        score = round(float(np.clip(raw_score, 0.0, 1.0)), 4)

        return DemandResponse(
            demand_score=score,
            demand_level=demand_label(score),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
