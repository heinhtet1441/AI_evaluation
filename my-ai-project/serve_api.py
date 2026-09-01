import os
from typing import List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import uvicorn

# Import internal modules
from idea_scoring import score_idea
import data_store

app = FastAPI(title="AI Evaluation Engine API")

# Setup OLLAMA URL
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "").rstrip("/")


class FeedbackRow(BaseModel):
  idea_text: str
  human_score: float


class BulkFeedbackPayload(BaseModel):
  rows: List[FeedbackRow]


@app.get("/health")
def health_check():
  return {"status": "ok", "model_version": "v1.0.0-NLP"}


@app.get("/feedback-count")
def get_feedback_count():
  count = data_store.get_total_feedback_count()
  return {"count": count}


@app.get("/metrics")
def get_metrics():
  return data_store.get_latest_metrics()


@app.get("/model-info")
def get_model_info():
  return {"feature_names": "N/A (TF-IDF Vectorizer Active)"}


@app.post("/score-idea")
async def score_idea_endpoint(
    text: Optional[str] = Form(None),
    use_llm: str = Form("true"),
    org_context: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
):
  use_llm_bool = str(use_llm).lower() == "true"
  extracted_text = ""

  if file:
    content = await file.read()
    # Simple text extraction fallback
    extracted_text = content.decode("utf-8", errors="ignore")
  elif text:
    extracted_text = text

  if not extracted_text.strip():
    raise HTTPException(
        status_code=400, detail="No text or file content provided"
    )

  try:
    # Unexpected keyword argument FIX: Only pass arguments supported by score_idea()
    result = score_idea(
        text=extracted_text,
        use_llm=use_llm_bool,
        org_context=org_context or "",
    )
    return result
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/bulk-feedback")
def bulk_feedback_endpoint(payload: BulkFeedbackPayload):
  try:
    added = data_store.save_bulk_feedback(
        [r.model_dump() for r in payload.rows]
    )
    return {"status": "success", "added": added}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  uvicorn.run(app, host="0.0.0.0", port=port)
