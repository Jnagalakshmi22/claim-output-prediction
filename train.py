"""Generate synthetic claims, train and evaluate a leakage-aware classifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

sys.modules.setdefault("train", sys.modules[__name__])

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "synthetic_claims.csv"
MODEL_FILE = ROOT / "claim_pipeline.joblib"
CLASSES = ["Paid", "Partially_Paid", "Denied"]
FEATURE_COLUMNS = [
    "payer",
    "provider",
    "cpt_code",
    "diagnosis_code",
    "billed_amount",
    "patient_age",
]
CATEGORICAL_COLUMNS = ["payer", "provider", "cpt_code", "diagnosis_code"]
NUMERIC_COLUMNS = [
    "billed_amount",
    "patient_age",
    "payer_denial_rate",
    "provider_denial_rate",
    "amount_to_age_ratio",
    "payer_amount_deviation",
]
CPT_AMOUNT_RANGES = {
    "99213": (100, 250),
    "80053": (100, 250),
    "99214": (220, 450),
    "99203": (220, 450),
    "99215": (350, 650),
    "73721": (600, 1500),
}
PAYER_RISK = {"Aetna": 0.0, "BlueCross": 0.05, "Humana": 0.1, "Cigna": 0.2, "UnitedHC": 0.25}
PROCEDURE_RISK = {
    "99213": 0.0,
    "80053": 0.0,
    "99214": 0.12,
    "99203": 0.12,
    "99215": 0.25,
    "73721": 0.4,
}
HIGHER_RISK_PROVIDERS = {"Dr. Shah", "Dr. Johnson"}


def generate_synthetic_claims(n_samples: int = 500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create reproducible claims with noisy outcomes influenced by claim attributes."""
    rng = np.random.default_rng(random_state)
    payers = np.array(["Aetna", "BlueCross", "Cigna", "UnitedHC", "Humana"])
    providers = np.array(["Dr. Rao", "Dr. Shah", "Dr. Lee", "Dr. Smith", "Dr. Johnson"])
    cpt_codes = np.array(["99213", "99214", "99215", "99203", "73721", "80053"])
    diagnoses = np.array(["M54.5", "J06.9", "E11.9", "I10", "R05"])
    records = []

    for claim_id in range(1, n_samples + 1):
        payer = str(rng.choice(payers, p=[0.25, 0.25, 0.20, 0.20, 0.10]))
        provider = str(rng.choice(providers))
        cpt = str(rng.choice(cpt_codes, p=[0.30, 0.25, 0.15, 0.10, 0.10, 0.10]))
        diagnosis = str(rng.choice(diagnoses))
        age = int(rng.integers(18, 85))
        low_amount, high_amount = CPT_AMOUNT_RANGES[cpt]
        billed_amount = round(float(rng.uniform(low_amount, high_amount)), 2)
        records.append(
            {
                "id": claim_id,
                "payer": payer,
                "provider": provider,
                "cpt_code": cpt,
                "diagnosis_code": diagnosis,
                "billed_amount": billed_amount,
                "patient_age": age,
            }
        )

    claims = pd.DataFrame.from_records(records)
    risk = calculate_synthetic_risk(claims).to_numpy()
    standardized_risk = (risk - risk.mean()) / risk.std()
    logits = np.column_stack(
        [
            np.zeros(n_samples),
            -1.82 + 2.8 * standardized_risk,
            -5.15 + 5.0 * standardized_risk,
        ]
    )
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    draws = rng.random(n_samples)
    class_indexes = (draws[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
    claims["status"] = np.asarray(CLASSES)[class_indexes]
    return claims


def calculate_synthetic_risk(frame: pd.DataFrame) -> pd.Series:
    """Latent risk used only to simulate noisy labels, never supplied to the model."""
    amount_ranges = frame["cpt_code"].map(
        lambda code: CPT_AMOUNT_RANGES.get(code, (100, 250))
    )
    amount_low = amount_ranges.map(lambda bounds: bounds[0])
    amount_high = amount_ranges.map(lambda bounds: bounds[1])
    amount_position = (
        (frame["billed_amount"] - amount_low) / (amount_high - amount_low)
    ).clip(0, 1)
    risk = (
        frame["payer"].map(PAYER_RISK).fillna(0.1)
        + frame["cpt_code"].map(PROCEDURE_RISK).fillna(0.1)
        + 0.2 * amount_position
        + 0.12 * frame["provider"].isin(HIGHER_RISK_PROVIDERS).astype(float)
        + 0.1 * (frame["patient_age"] > 70).astype(float)
        + 0.08 * frame["diagnosis_code"].isin({"E11.9", "I10"}).astype(float)
    )
    return risk.astype(float)


class LeakageFreeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Learn training aggregates and exclude each row's own label while fitting."""

    __module__ = "train"

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        if y is None:
            raise ValueError("Target values are required to learn denial-rate features.")
        frame = X.copy()
        denied = pd.Series(np.asarray(y) == "Denied", index=frame.index, dtype=float)
        self.global_denial_rate_ = float(denied.mean())
        self.payer_denial_rates_ = denied.groupby(frame["payer"]).mean().to_dict()
        self.provider_denial_rates_ = denied.groupby(frame["provider"]).mean().to_dict()
        self.global_avg_amount_ = float(frame["billed_amount"].mean())
        self.payer_avg_amount_ = frame.groupby("payer")["billed_amount"].mean().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        frame["payer_denial_rate"] = (
            frame["payer"].map(self.payer_denial_rates_).fillna(self.global_denial_rate_)
        )
        frame["provider_denial_rate"] = (
            frame["provider"].map(self.provider_denial_rates_).fillna(self.global_denial_rate_)
        )
        frame["amount_to_age_ratio"] = frame["billed_amount"] / frame["patient_age"].clip(lower=1)
        payer_average = frame["payer"].map(self.payer_avg_amount_).fillna(self.global_avg_amount_)
        frame["payer_amount_deviation"] = frame["billed_amount"] - payer_average
        return frame

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        if y is None:
            raise ValueError("Target values are required to learn denial-rate features.")
        self.fit(X, y)
        frame = X.copy()
        denied = pd.Series(
            np.asarray(y) == "Denied", index=frame.index, dtype=float
        )
        global_rate = (denied.sum() - denied) / max(len(denied) - 1, 1)

        for column, output_column in (
            ("payer", "payer_denial_rate"),
            ("provider", "provider_denial_rate"),
        ):
            grouped = denied.groupby(frame[column])
            sums = grouped.transform("sum")
            counts = grouped.transform("count")
            fallback = global_rate
            frame[output_column] = ((sums - denied) / (counts - 1)).where(
                counts > 1, fallback
            )

        amount_sum = frame.groupby("payer")["billed_amount"].transform("sum")
        amount_count = frame.groupby("payer")["billed_amount"].transform("count")
        global_amount = (
            frame["billed_amount"].sum() - frame["billed_amount"]
        ) / max(len(frame) - 1, 1)
        payer_average = ((amount_sum - frame["billed_amount"]) / (amount_count - 1)).where(
            amount_count > 1, global_amount
        )
        frame["amount_to_age_ratio"] = frame["billed_amount"] / frame["patient_age"].clip(lower=1)
        frame["payer_amount_deviation"] = frame["billed_amount"] - payer_average
        return frame


def build_pipeline() -> Pipeline:
    """Build the complete preprocessing and classification pipeline."""
    preprocessing = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )
    return Pipeline(
        [
            ("feature_engineer", LeakageFreeFeatureEngineer()),
            ("preprocessor", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    C=0.05,
                    max_iter=1000,
                ),
            ),
        ]
    )


def train_and_evaluate() -> dict:
    """Generate data, stratify, fit using training rows, evaluate, and persist."""
    claims = generate_synthetic_claims()
    claims.to_csv(DATA_FILE, index=False)
    X = claims[FEATURE_COLUMNS]
    y = claims["status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    report = classification_report(
        y_test, predictions, labels=CLASSES, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=CLASSES)
    temporary_model_file = MODEL_FILE.with_suffix(".joblib.tmp")
    joblib.dump(pipeline, temporary_model_file)
    os.replace(temporary_model_file, MODEL_FILE)
    metrics = {
        "class_order": CLASSES,
        "test_size": len(y_test),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "support": y_test.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
    }
    print("Test classification report:")
    print(classification_report(y_test, predictions, labels=CLASSES, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; order: Paid, Partially_Paid, Denied):")
    print(matrix)
    print(f"\nSaved {len(claims)} rows to {DATA_FILE.name} and model to {MODEL_FILE.name}.")
    print("Metrics JSON:")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_and_evaluate()
