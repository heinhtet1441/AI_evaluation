import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import data_store as ds

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_idea_brain")

ARTIFACT_ROOT = "idea_model_artifacts"
MIN_ROWS_TO_TRAIN = 20
SYNTHETIC_DATA_PATH = "mock_idea_train_data.csv"

def load_feedback() -> pd.DataFrame:
    df = pd.DataFrame()
    if os.path.exists(SYNTHETIC_DATA_PATH):
        try:
            df = pd.read_csv(SYNTHETIC_DATA_PATH)
            col_map = {}
            for col in df.columns:
                if col.lower() in ["text", "idea_text", "proposal", "description"]:
                    col_map[col] = "idea_text"
                elif col.lower() in ["score", "human_score", "rating", "label"]:
                    col_map[col] = "human_score"
            df = df.rename(columns=col_map).dropna(subset=["idea_text"])
        except Exception as e:
            logger.warning("Error reading %s: %s", SYNTHETIC_DATA_PATH, e)

    try:
        ds.init_db()
        db_df = ds.get_feedback_df()
        if db_df is not None and not db_df.empty:
            db_df = db_df.dropna(subset=["idea_text", "human_score"])
            if not db_df.empty:
                df = pd.concat([df, db_df], ignore_index=True)
    except Exception as e:
        logger.warning("Could not merge DB feedback: %s", e)

    if len(df) < MIN_ROWS_TO_TRAIN:
        raise FileNotFoundError(f"Insufficient data (found {len(df)} rows). Need at least {MIN_ROWS_TO_TRAIN}.")

    if "human_score" not in df.columns or df["human_score"].isnull().all() or df["human_score"].std() < 0.1:
        scores = []
        for text in df["idea_text"]:
            words = str(text).split()
            word_count = len(words)
            digits = sum(c.isdigit() for c in str(text))
            caps = sum(1 for w in words if w.isupper() and len(w) > 1)
            sentences = [s for s in str(text).split('.') if s.strip()]
            avg_sent_len = word_count / max(len(sentences), 1)
            clarity = max(1.0, 10.0 - max(0, avg_sent_len - 15) * 0.3)
            length_score = 10.0 if 80 <= word_count <= 400 else max(1.0, 10.0 - (abs(word_count - 240) / 40.0))
            specificity_score = min(10.0, (digits * 0.4) + (caps * 0.2) + 3.5)
            final_s = np.clip((clarity * 0.3) + (length_score * 0.4) + (specificity_score * 0.3), 1.0, 10.0)
            scores.append(round(final_s, 2))
        df["human_score"] = scores

    return df

def train(df: pd.DataFrame):
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["idea_text"], df["human_score"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,  # Optimized for performance
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Parallel Execution Enabled (n_jobs=-1 for faster runtime)
    model_ridge = Ridge(alpha=0.8)
    model_et = ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model_gb = GradientBoostingRegressor(n_estimators=80, learning_rate=0.08, max_depth=4, random_state=42)

    model = VotingRegressor(
        estimators=[("ridge", model_ridge), ("et", model_et), ("gb", model_gb)],
        weights=[1.0, 1.5, 1.5],
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 1.0, 10.0)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "n_training_rows": len(df),
    }

    return vectorizer, model, metrics

def save_brain(vectorizer, model, metrics):
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(ARTIFACT_ROOT, version)
    os.makedirs(version_dir, exist_ok=True)

    joblib.dump(vectorizer, os.path.join(version_dir, "vectorizer.joblib"))
    joblib.dump(model, os.path.join(version_dir, "model.joblib"))
    with open(os.path.join(version_dir, "metrics.json"), "w") as f:
        json.dump({**metrics, "trained_at": version}, f, indent=2)

    with open(os.path.join(ARTIFACT_ROOT, "latest.txt"), "w") as f:
        f.write(version_dir)

    # Save tracking history directly into Database
    ds.register_model_version(version_dir, metrics)

    return version_dir

def run_training() -> dict:
    df = load_feedback()
    vectorizer, model, metrics = train(df)
    version_dir = save_brain(vectorizer, model, metrics)
    return {"version_dir": version_dir, "metrics": metrics}

if __name__ == "__main__":
    run_training()