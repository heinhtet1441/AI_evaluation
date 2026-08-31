"""
staff_admin_dashboard.py
========================
Internal Staff & Admin Portal for AI Training, Data Ingestion, & System Health.
"""

import os
import pandas as pd
import requests
import streamlit as st

# Cloud Deployment & Local Dynamic API URL Configuration
API_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="AI Lab Admin Portal", layout="wide")
st.title("⚡ AI Lab Admin Portal — Model Retraining & Research Ingestion")

# Header & Health Check
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    model_version = health.get("model_version", "N/A")
    st.sidebar.success(f"Connected to API Core\nVersion: {model_version}")
except Exception:
    st.sidebar.error("Cannot connect to Backend API Engine.")
    st.warning(f"⚠️ Backend API Server မပွင့်သေးပါ သို့မဟုတ် URL လွဲနေပါသည်။ (`{API_URL}`)")

tab_train_brain, tab_metrics, tab_predict = st.tabs([
    "Train AI Brain (Open Data / Docs Ingestion)",
    "Model Evaluation Metrics",
    "Numeric Model Testing",
])

# ---------------------------------------------------------------------------
# Tab 1: Train AI Brain with Open-Source Data / CSVs / Research Docs
# ---------------------------------------------------------------------------
with tab_train_brain:
    st.subheader("Train Brain on Research Docs, Open-Source Datasets & Feedback")

    try:
        resp = requests.get(f"{API_URL}/feedback-count", timeout=5)
        if resp.status_code == 200:
            fb_count = resp.json().get("count", 0)
            st.info(f"Total Training Samples in DB: **{fb_count}** (At least 20 samples required to retrain)")
    except Exception:
        pass

    st.markdown("### 1. Ingest Open-Source CSV Datasets")
    st.caption("Import CSVs with `idea_text` and `human_score` (0-10) columns.")
    csv_upload = st.file_uploader("Upload Open-Source CSV Dataset", type=["csv"], key="staff_csv")
    if csv_upload is not None:
        try:
            train_df = pd.read_csv(csv_upload)
            if "idea_text" in train_df.columns and "human_score" in train_df.columns:
                st.dataframe(train_df.head(5))
                if st.button("Import CSV to Training DB"):
                    rows = [
                        {"idea_text": str(r["idea_text"]), "human_score": float(r["human_score"])}
                        for _, r in train_df.iterrows()
                    ]
                    resp = requests.post(f"{API_URL}/bulk-feedback", json={"rows": rows}, timeout=120)
                    if resp.status_code == 200:
                        st.success(f"Successfully added {resp.json().get('added', len(rows))} samples to Database!")
                    else:
                        st.error(f"Failed to import CSV: {resp.text}")
            else:
                st.error("CSV file must contain `idea_text` and `human_score` columns.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    st.divider()
    st.markdown("### 2. Ingest Research Papers & Open Docs (.pdf, .docx, .txt)")
    st.caption("Upload multiple research papers/documents at once. Text is parsed automatically.")

    docs_uploaded = st.file_uploader(
        "Upload Research Files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="staff_docs",
    )
    if "admin_docs" not in st.session_state:
        st.session_state["admin_docs"] = {}

    if docs_uploaded:
        for f in docs_uploaded:
            if f.name not in st.session_state["admin_docs"]:
                try:
                    resp = requests.post(
                        f"{API_URL}/extract-text",
                        files={"file": (f.name, f.getvalue(), f.type)},
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        st.session_state["admin_docs"][f.name] = {
                            "full_text": resp.json().get("extracted_text", ""),
                            "score": 5.0,
                        }
                    else:
                        st.error(f"Failed to extract text from {f.name}: {resp.text}")
                except Exception as e:
                    st.error(f"Error connecting for {f.name}: {e}")

    if st.session_state["admin_docs"]:
        preview_rows = [
            {"filename": k, "preview": v["full_text"][:120] + "...", "score": v["score"]}
            for k, v in st.session_state["admin_docs"].items()
        ]
        edited = st.data_editor(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)
        for _, row in edited.iterrows():
            st.session_state["admin_docs"][row["filename"]]["score"] = row["score"]

        if st.button("Add Documents to Training DB"):
            rows = [
                {"idea_text": v["full_text"], "human_score": float(v["score"])}
                for v in st.session_state["admin_docs"].values()
            ]
            resp = requests.post(f"{API_URL}/bulk-feedback", json={"rows": rows}, timeout=120)
            if resp.status_code == 200:
                st.success("Research documents successfully added to Database!")
                st.session_state["admin_docs"] = {}
                st.rerun()
            else:
                st.error(f"Failed to save documents to DB: {resp.text}")

    st.divider()
    st.markdown("### 3. Retrain AI Model Now")
    if st.button("Trigger Retrain & Hot-Reload Model", type="primary"):
        with st.spinner("Training Model across datasets..."):
            try:
                resp = requests.post(f"{API_URL}/train-idea-brain", timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    m = data.get("metrics", {})
                    st.success(f"Model successfully Retrained & Deployed Live! (Version: `{data.get('version_dir', 'N/A')}`)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE", f"{m.get('MAE', 0.0):.3f}")
                    c2.metric("RMSE", f"{m.get('RMSE', 0.0):.3f}")
                    c3.metric("R² Score", f"{m.get('R2', 0.0):.3f}")
                else:
                    st.error(f"Retraining failed: {resp.text}")
            except Exception as e:
                st.error(f"Training connection error: {e}")

# ---------------------------------------------------------------------------
# Tab 2 & 3: Model Metrics & Numeric Testing
# ---------------------------------------------------------------------------
with tab_metrics:
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        if resp.status_code == 200:
            metrics = resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{metrics.get('MAE', 0.0):.4f}")
            c2.metric("MSE", f"{metrics.get('MSE', 0.0):.4f}")
            c3.metric("RMSE", f"{metrics.get('RMSE', 0.0):.4f}")
            c4.metric("R²", f"{metrics.get('R2', 0.0):.4f}")
        else:
            st.warning("No metrics available yet. Retrain model to generate initial metrics.")
    except Exception as e:
        st.warning(f"Unable to load metrics: {e}")

with tab_predict:
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            st.write("Feature Names Expected:", info.get("feature_names", "N/A (NLP Vectorizer Used)"))
        else:
            st.info("NLP Vectorizer-based Model is active. (No manual features required)")
    except Exception as e:
        st.info("NLP Vectorizer-based Model is active.")