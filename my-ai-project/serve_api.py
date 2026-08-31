import io
import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

from idea_scoring import score_idea, reload_learned_brain, PILLARS, extract_text
import data_store as ds
import train_idea_brain

LLM_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEFAULT_ORG_CONTEXT = os.environ.get("ORG_CONTEXT")
ARTIFACT_ROOT = "idea_model_artifacts"

_state = {"model": None, "vectorizer": None, "metrics": None, "version_dir": None}

def load_latest():
    latest_pointer = os.path.join(ARTIFACT_ROOT, "latest.txt")
    if not os.path.exists(latest_pointer):
        return

    with open(latest_pointer) as f:
        version_dir = f.read().strip()

    model_path = os.path.join(version_dir, "model.joblib")
    vectorizer_path = os.path.join(version_dir, "vectorizer.joblib")
    metrics_path = os.path.join(version_dir, "metrics.json")

    if os.path.exists(model_path):
        _state["model"] = joblib.load(model_path)
    if os.path.exists(vectorizer_path):
        _state["vectorizer"] = joblib.load(vectorizer_path)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _state["metrics"] = json.load(f)

    _state["version_dir"] = version_dir

@asynccontextmanager
async def lifespan(app: FastAPI):
    ds.init_db()
    load_latest()
    yield

app = FastAPI(title="AI Evaluation Model API", lifespan=lifespan)

# Allow Cross-Origin Requests for Cloud Deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def perform_ocr_extraction(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    extracted_text = ""
    try:
        if ext == "pdf":
            images = convert_from_bytes(file_bytes)
            for page_num, image in enumerate(images):
                ocr_text = pytesseract.image_to_string(image)
                extracted_text += f"\n--- Page {page_num + 1} (OCR) ---\n" + ocr_text
        elif ext in ["png", "jpg", "jpeg", "tiff", "bmp"]:
            image = Image.open(io.BytesIO(file_bytes))
            extracted_text = pytesseract.image_to_string(image)
        else:
            extracted_text = extract_text(file_bytes=file_bytes, filename=filename)
    except Exception:
        extracted_text = extract_text(file_bytes=file_bytes, filename=filename)

    return extracted_text.strip()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": _state.get("version_dir", "unloaded")}

@app.get("/metrics")
def metrics():
    if _state["metrics"] is None:
        raise HTTPException(status_code=404, detail="No metrics available.")
    return _state["metrics"]

@app.get("/model-history")
def model_history():
    df = ds.get_all_model_versions()
    return df.to_dict(orient="records")

@app.post("/score-idea")
def score_idea_endpoint(
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    use_llm: bool = Form(False),
    org_context: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    provided = [v for v in (text, url, file) if v]
    if len(provided) != 1:
        raise HTTPException(status_code=400, detail="Provide exactly one of: text, url, or file.")

    file_bytes, filename = None, None
    if file is not None:
        file_bytes = file.file.read()
        filename = file.filename

    source_type = "text" if text else ("file" if file is not None else "url")

    try:
        result = score_idea(
            text=text,
            file_bytes=file_bytes,
            filename=filename,
            url=url,
            reference_corpus=ds.get_reference_corpus() or None,
            use_llm=use_llm,
            api_key=LLM_API_KEY if use_llm else None,
            org_context=org_context or DEFAULT_ORG_CONTEXT,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    ds.save_submission(source_type, result)
    return result

@app.post("/train-idea-brain")
def train_idea_brain_endpoint():
    try:
        result = train_idea_brain.run_training()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    reload_learned_brain()
    load_latest()
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve_api:app", host="0.0.0.0", port=8000)