# ml-pipeline/inference.py
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, recall_score

from .config import TARGET_RECALL_R


def save_model(
    model,
    threshold: float,
    features: list[str],
    model_name: str = "amr_classifier",
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{model_name}_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "features": features,
            "created_at": timestamp,
        },
        model_path,
    )
    print(f"💾 Saved: {model_path}")
    return model_path


def load_model(model_path: str) -> tuple:
    """Load model từ file joblib. Returns (model, threshold, features)."""
    data = joblib.load(model_path)
    return data["model"], data["threshold"], data["features"]


def find_best_threshold_inner(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_recall: float = TARGET_RECALL_R,
) -> tuple[float, float]:
    thresholds = np.arange(0.30, 0.71, 0.001)
    rows = []
    for th in thresholds:
        y_p = (y_proba >= th).astype(int)
        rows.append({
            "threshold": th,
            "macro_f1": f1_score(y_true, y_p, average="macro"),
            "recall_R": recall_score(y_true, y_p, pos_label=1),
        })
    df = pd.DataFrame(rows)
    candidates = df[df["recall_R"] >= target_recall]
    if len(candidates) > 0:
        best = candidates.sort_values("macro_f1", ascending=False).iloc[0]
    else:
        best = df.sort_values("macro_f1", ascending=False).iloc[0]
    return float(best["threshold"]), float(best["macro_f1"])


def predict_one_patient(
    feature_vector: pd.Series,
    model,
    threshold: float,
    expected_features: list[str],
) -> dict:

    X = feature_vector.reindex(expected_features).values.reshape(1, -1)
    proba = model.predict_proba(X)[0, 1]
    prediction = "Resistant" if proba >= threshold else "Susceptible"
    return {
        "prediction": prediction,
        "prob_resistant": round(float(proba), 4),
        "confidence": round(float(max(proba, 1 - proba)), 4),
        "threshold_used": threshold,
    }