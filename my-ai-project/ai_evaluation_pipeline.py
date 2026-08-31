"""
AI Model Evaluation Pipeline
============================
Architecture: Data -> Preprocess -> Train/Val/Test -> Model -> Multi-Metric
              Evaluation -> Explainability -> Drift-Monitoring Hook -> Save

This upgrades the original quick script into a structure that follows the
core pillars of a real AI evaluation system:
  1. Performance metrics      -> MAE / MSE / RMSE / R2 (not just loss)
  2. Robustness                -> proper train/val/test split + early stopping
  3. Explainability             -> SHAP feature importance
  4. Continuous monitoring      -> PSI-based drift check stub for production
  5. Reproducibility & logging  -> logging module + fixed seeds + versioned save
"""

import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------------------------------------------------------
# 0. Reproducibility & Logging
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ai_eval_pipeline")


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str = "company_numeric_data.csv") -> pd.DataFrame:
    """
    Attempts to read a numeric training CSV with columns matching your
    features plus a 'Target' column. If the file does not exist, falls back
    to generating synthetic data so the script still runs end-to-end.

    NOTE: this is the NUMERIC regression pipeline (features -> Target),
    unrelated to mock_idea_train_data.csv (which feeds the separate text-based
    idea-scoring brain in train_idea_brain.py — different schema entirely,
    idea_text/human_score, not A-E/Target). Do not point this at that file.
    """
    if os.path.exists(csv_path):
        logger.info("Loading data from '%s'", csv_path)
        return pd.read_csv(csv_path)

    logger.warning("File '%s' not found. Generating synthetic fallback dataset.", csv_path)
    data = np.random.rand(100, 6)
    df = pd.DataFrame(data, columns=["A", "B", "C", "D", "E", "Target"])
    return df


# ---------------------------------------------------------------------------
# 2. Data Validation & Cleaning
#    (a real evaluation system checks the INPUT data quality first,
#     not just the model's output)
# ---------------------------------------------------------------------------
def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    if n_before != n_after:
        logger.warning("Dropped %d rows containing missing values", n_before - n_after)

    # basic sanity checks — catch obviously broken data before it reaches the model
    if df.isnull().values.any():
        raise ValueError("Data still contains nulls after cleaning.")
    if not np.isfinite(df.select_dtypes(include=[np.number]).values).all():
        raise ValueError("Data contains inf/-inf values.")

    logger.info("Data validated: %d rows, %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. Split: train / validation / test
#    Validation set is what was MISSING in the original script — without it
#    you have no way to detect overfitting during training.
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame, target_col: str = "Target"):
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # 60% train / 20% val / 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 4. Preprocessing
#    Scaler is fit ONLY on train data, then persisted so the exact same
#    transform can be applied to new data in production (avoids train/serve
#    skew, a common source of silent model degradation).
# ---------------------------------------------------------------------------
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# 5. Model Architecture
#    Added: BatchNorm (stabilizes training), Dropout (reduces overfitting).
#    Depth/width are still modest — tune based on your real dataset size.
# ---------------------------------------------------------------------------
def build_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# ---------------------------------------------------------------------------
# 6. Training
#    EarlyStopping + ReduceLROnPlateau give you the "robustness testing"
#    pillar almost for free: the model stops before it overfits, and the
#    learning rate backs off automatically when progress stalls.
# ---------------------------------------------------------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    logger.info("Training finished after %d epochs (best weights restored)", len(history.history["loss"]))
    return history


# ---------------------------------------------------------------------------
# 7. Multi-Metric Evaluation
#    A single "loss" number hides a lot. R2 tells you how much variance the
#    model actually explains; RMSE is in the same units as your target,
#    which is what stakeholders actually understand.
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test, verbose=0).flatten()

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
    }
    for name, value in metrics.items():
        logger.info("Test %s: %.4f", name, value)
    return metrics


# ---------------------------------------------------------------------------
# 8. Explainability
#    SHAP tells you which of A-E actually drives predictions — required if
#    this model's output will inform a business/ethical decision.
#    Wrapped in try/except so the pipeline doesn't hard-fail if shap isn't
#    installed yet (pip install shap --break-system-packages).
# ---------------------------------------------------------------------------
def explain_model(model, X_train, X_test, feature_names):
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed — skipping explainability step. "
                       "Install with: pip install shap --break-system-packages")
        return None

    background = X_train[np.random.choice(X_train.shape[0], min(50, X_train.shape[0]), replace=False)]
    explainer = shap.Explainer(model, background)
    shap_values = explainer(X_test[:20])  # sample for speed

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0).flatten()
    importance = sorted(zip(feature_names, mean_abs_shap.tolist()), key=lambda x: -x[1])
    logger.info("Feature importance (mean |SHAP value|): %s", importance)
    return importance


# ---------------------------------------------------------------------------
# 9. Continuous Monitoring Hook (production stub)
#    Population Stability Index compares the distribution of a NEW batch of
#    incoming data against the training distribution. A high PSI means the
#    data has drifted and the model likely needs retraining/investigation.
#    Call this periodically in production with fresh data, not at train time.
# ---------------------------------------------------------------------------
def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(expected, breakpoints)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=bin_edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bin_edges)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)  # < 0.1 stable | 0.1-0.25 moderate shift | > 0.25 significant drift


# ---------------------------------------------------------------------------
# 10. Save model + scaler with a version stamp
#     Versioning is what lets you roll back if a new model underperforms.
# ---------------------------------------------------------------------------
def save_artifacts(model, scaler, feature_names, metrics, out_dir="model_artifacts"):
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(out_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    model.save(os.path.join(version_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(version_dir, "scaler.joblib"))

    # feature_names + metrics are what the API and dashboard read —
    # keeping them alongside the model means every version is self-describing.
    with open(os.path.join(version_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump({**metrics, "trained_at": version}, f, indent=2)

    # "latest" pointer file — the API always reads this to find the current model
    with open(os.path.join(out_dir, "latest.txt"), "w") as f:
        f.write(version_dir)

    logger.info("Saved model + scaler to %s", version_dir)
    return version_dir


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    df = validate_and_clean(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    feature_names = X_train.columns.tolist()

    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    model = build_model(input_dim=X_train_s.shape[1])
    train_model(model, X_train_s, y_train, X_val_s, y_val)

    metrics = evaluate_model(model, X_test_s, y_test)
    explain_model(model, X_train_s, X_test_s, feature_names)

    # Example of the drift check you'd run periodically in production,
    # comparing training data to a fresh incoming batch:
    drift_score = population_stability_index(X_train_s[:, 0], X_test_s[:, 0])
    logger.info("Example PSI (feature 0, train vs test): %.4f", drift_score)

    save_artifacts(model, scaler, feature_names, metrics)
    return metrics


if __name__ == "__main__":
    main()