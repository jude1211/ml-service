"""
train_model.py
--------------
Train an XGBoost regression model to predict cinema show demand (0-1).

Features:
  - show_hour          : int  (0-23)  Hour of day the show starts
  - day_of_week        : int  (0-6)   0=Monday, 6=Sunday
  - seat_occupancy_pct : float (0-1)  Current % of seats already booked
  - movie_popularity   : float (0-1)  Normalised IMDB/rating score
  - recent_bookings    : int         Number of bookings in last 60 min

Output:
  - demand_score : float (0-1)

Usage:
  python train_model.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "cinema_demand.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.pkl")

FEATURES = [
    "show_hour",
    "day_of_week",
    "seat_occupancy_pct",
    "movie_popularity",
    "recent_bookings",
]
TARGET = "demand_score"

# ── Load data ──────────────────────────────────────────────────────────────────
print("[*] Loading dataset ...")
df = pd.read_csv(DATA_PATH)
print(f"    Rows: {len(df)} | Columns: {list(df.columns)}")

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ── Model ──────────────────────────────────────────────────────────────────────
print("[*] Training XGBoost model ...")
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
# Clip predictions to [0, 1] range
y_pred = np.clip(y_pred, 0.0, 1.0)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
print(f"[*] Test MAE  : {mae:.4f}")
print(f"[*] Test R²   : {r2:.4f}")

# ── Save model ─────────────────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"[*] Model saved -> {MODEL_PATH}")

# ── Quick sanity check ─────────────────────────────────────────────────────────
print("\n[*] Sample predictions (first 5 test rows):")
sample = X_test.head(5).copy()
sample["actual"]    = y_test.values[:5]
sample["predicted"] = np.clip(model.predict(X_test.head(5)), 0, 1).round(3)
print(sample.to_string(index=False))
