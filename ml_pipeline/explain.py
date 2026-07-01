# ml-pipeline/explain.py
import numpy as np
import pandas as pd
import shap

def build_shap_explainer(model, X_background: pd.DataFrame):
    explainer = shap.Explainer(model.predict_proba, X_background)
    return explainer


def explain_prediction(
    explainer,
    feature_vector: pd.Series,
    expected_features: list[str],
    top_k: int = 10,
) -> dict:
    X = feature_vector.reindex(expected_features).values.reshape(1, -1)
    num_features = X.shape[1]
    shap_values = explainer(X, max_evals=max(2 * num_features + 1, 1000))
    vals = shap_values.values[0, :, 1]
    cols = np.array(expected_features)

    top_idx = np.argsort(vals)[::-1][:top_k]
    top_features = [
        {"feature": str(cols[i]),
         "shap_value": round(float(vals[i]), 4),
         "feature_value": round(float(X[0, i]), 4)}
        for i in top_idx
    ]
    return {
        "top_features": top_features,
        "base_value": round(float(shap_values.base_values[0, 1]), 4),
        "prediction_value": round(float(vals.sum() + shap_values.base_values[0, 1]), 4),
    }
