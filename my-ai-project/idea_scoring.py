import io
import os
import re
import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

IDEA_MODEL_ROOT = "idea_model_artifacts"

# Dynamic Base URL resolution
def _get_ollama_base_url() -> str:
    url = os.environ.get("OLLAMA_BASE_URL", "https://boost-front-generous-johns.trycloudflare.com").rstrip("/")
    return url

def clean_and_normalize_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = BeautifulSoup(raw_text, "html.parser").get_text(separator=" ")
    text = text.replace('\xa0', ' ').replace('\r', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text(text: str = None, file_bytes: bytes = None, filename: str = None, url: str = None) -> str:
    extracted = ""
    if text:
        extracted = text
    elif file_bytes is not None and filename:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "pdf":
            extracted = _extract_pdf(file_bytes)
        elif ext == "docx":
            extracted = _extract_docx(file_bytes)
        elif ext == "txt":
            extracted = file_bytes.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: .{ext}")
    elif url:
        extracted = _extract_url(url)
    else:
        raise ValueError("Provide one of: text, file_bytes+filename, or url.")

    return clean_and_normalize_text(extracted)

def _extract_pdf(file_bytes: bytes) -> str:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def _extract_docx(file_bytes: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs)

def _extract_url(url: str) -> str:
    resp = requests.get(url, timeout=10, headers={"User-Agent": "FastAPI-Backend-Engine/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main")
    return main.get_text(separator="\n") if main else "\n".join(p.get_text() for p in soup.find_all("p"))

@dataclass
class HeuristicResult:
    clarity: float
    specificity: float
    length_adequacy: float
    novelty: float
    heuristic_overall: float
    details: dict = field(default_factory=dict)

def heuristic_scores(idea_text: str, reference_corpus: list[str] = None) -> HeuristicResult:
    words = idea_text.split()
    sentences = [s for s in re.split(r"[.!?]+", idea_text) if s.strip()]
    n_words = len(words)
    n_sentences = max(len(sentences), 1)
    avg_sentence_len = n_words / n_sentences

    clarity = max(0, 10 - max(0, avg_sentence_len - 20) * 0.3)
    clarity = min(clarity, 10)

    digit_count = sum(c.isdigit() for c in idea_text)
    capitalized = sum(1 for w in words if w[:1].isupper())
    specificity = min(10, (digit_count / max(n_words, 1) * 200) + (capitalized / max(n_words, 1) * 20))

    ideal_min, ideal_max = 50, 600
    if n_words < ideal_min:
        length_adequacy = 10 * (n_words / ideal_min)
    elif n_words > ideal_max:
        length_adequacy = max(0, 10 - (n_words - ideal_max) / 200)
    else:
        length_adequacy = 10

    if reference_corpus and len(reference_corpus) > 0:
        corpus = reference_corpus + [idea_text]
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(corpus)
        sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        max_sim = sims.max() if len(sims) else 0
        novelty = round((1 - max_sim) * 10, 2)
    else:
        novelty = 5.0

    heuristic_overall = round(
        clarity * 0.25 + specificity * 0.25 + length_adequacy * 0.25 + novelty * 0.25, 2
    )

    return HeuristicResult(
        clarity=round(clarity, 2),
        specificity=round(specificity, 2),
        length_adequacy=round(length_adequacy, 2),
        novelty=novelty,
        heuristic_overall=heuristic_overall,
        details={"word_count": n_words, "sentence_count": n_sentences, "avg_sentence_length": round(avg_sentence_len, 1)},
    )

PILLARS = [
    {"key": "feasibility", "name": "Feasibility (general)", "measures": "A quick, general read on buildability.", "focus": "Is this plausible?", "weight": 1.0},
    {"key": "market_potential", "name": "Market Potential (general)", "measures": "General read on opportunity size.", "focus": "Is there a market?", "weight": 1.0},
    {"key": "innovation", "name": "Innovation (general)", "measures": "General read on creativity.", "focus": "Does it feel inventive?", "weight": 1.0},
    {"key": "risk", "name": "Risk (general)", "measures": "General read on downside risk.", "focus": "How risky overall?", "weight": 1.0},
    {"key": "novelty_differentiation", "name": "Novelty & Differentiation", "measures": "Uniqueness compared to existing solutions.", "focus": "Is it fundamentally new?", "weight": 1.0},
    {"key": "technical_feasibility", "name": "Technical & Operational Feasibility", "measures": "Practical ability to build given technology constraints.", "focus": "Can this be engineered today?", "weight": 1.0},
    {"key": "value_creation_market_impact", "name": "Value Creation & Market Impact", "measures": "Magnitude of positive outcome if successful.", "focus": "How severe is the pain point solved?", "weight": 1.0},
    {"key": "problem_definition_groundedness", "name": "Problem Definition & Groundedness", "measures": "Logical soundness and real-world evidence.", "focus": "Is it based on real-world evidence?", "weight": 1.0},
    {"key": "scalability_growth_potential", "name": "Scalability & Growth Potential", "measures": "Workload growth without linear cost increase.", "focus": "Do unit economics improve with scale?", "weight": 1.0},
    {"key": "risk_safety_compliance", "name": "Risk, Safety & Compliance", "measures": "Legal, regulatory, and safety hurdles.", "focus": "Failure modes and regulatory risks.", "weight": 1.0},
    {"key": "resource_cost_efficiency", "name": "Resource & Cost Efficiency", "measures": "Capital, compute, and talent required.", "focus": "Is ROI justified?", "weight": 1.0},
    {"key": "adoption_friction_usability", "name": "Adoption Friction & User Usability", "measures": "Ease of integration into workflows.", "focus": "User workflow friction.", "weight": 1.0},
    {"key": "environmental_societal_sustainability", "name": "Environmental & Societal Sustainability", "measures": "Ethical and environmental impacts.", "focus": "Carbon/societal externalities.", "weight": 1.0},
    {"key": "strategic_goal_alignment", "name": "Strategic & Goal Alignment", "measures": "Alignment with company OKRs and vision.", "focus": "Fits organizational roadmap?", "weight": 1.0},
]
PILLAR_KEYS = [p["key"] for p in PILLARS]

def _build_rubric_prompt(idea_text: str, org_context: str = None) -> str:
    pillar_lines = "\n".join(
        f'- {p["name"]} ("{p["key"]}"): {p["measures"]} Focus: {p["focus"]}'
        for p in PILLARS
    )
    json_shape = ",\n".join(
        f'  "{p["key"]}": {{"score": <int>, "justification": "<text>"}}' for p in PILLARS
    )
    context_block = f'\nOrganization context:\n"""\n{org_context}\n"""\n' if org_context else "\nNo org context provided.\n"
    return f"""You are an enterprise innovation judge evaluating a proposal across 14 dimensions.
Score EACH dimension from 1 (very weak) to 10 (excellent).

SPECIAL INSTRUCTION FOR UNSTATED/MISSING CONTEXT:
If context is not explicitly specified, default the score to 7.0/10 and state context was unstated.

{pillar_lines}
{context_block}
Respond ONLY with a valid JSON object matching this shape:

{{
{json_shape}
}}

Idea Text:
\"\"\"
{idea_text[:6000]}
\"\"\"
"""

def llm_rubric_score(idea_text: str, api_key: str = None, model: str = "llama3", org_context: str = None, ollama_base_url: str = None) -> dict:
    prompt = _build_rubric_prompt(idea_text, org_context)
    
    clean_base_url = (ollama_base_url or _get_ollama_base_url()).rstrip("/")
    if clean_base_url.endswith("/v1"):
        clean_base_url = clean_base_url[:-3].rstrip("/")

    headers = {
        "User-Agent": "FastAPI-Backend-Engine/1.0",
        "Content-Type": "application/json"
    }

    # Attempt 1: Native Ollama /api/generate endpoint
    native_endpoint = f"{clean_base_url}/api/generate"
    native_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        resp = requests.post(native_endpoint, json=native_payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            raw_response = resp.json().get("response", "{}")
            rubric = json.loads(raw_response)
            for key in PILLAR_KEYS:
                if key not in rubric or not isinstance(rubric[key], dict) or "score" not in rubric[key]:
                    rubric[key] = {"score": 7.0, "justification": "Evaluated neutrally by Local Ollama Engine."}
            return rubric
        else:
            fallback_msg = f"API returned status code {resp.status_code}"
    except Exception as e:
        fallback_msg = str(e)

    # Attempt 2: OpenAI-compatible /v1/chat/completions endpoint
    chat_endpoint = f"{clean_base_url}/v1/chat/completions"
    chat_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI evaluation engine. Output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    try:
        resp = requests.post(chat_endpoint, json=chat_payload, headers=headers, timeout=60)
        if resp.status_code == 200:
            res_data = resp.json()
            raw_response = res_data["choices"][0]["message"]["content"]
            rubric = json.loads(raw_response)
            for key in PILLAR_KEYS:
                if key not in rubric or not isinstance(rubric[key], dict) or "score" not in rubric[key]:
                    rubric[key] = {"score": 7.0, "justification": "Evaluated neutrally by Local Ollama Engine."}
            return rubric
        else:
            fallback_msg = f"API returned status code {resp.status_code}"
    except Exception as e:
        fallback_msg = str(e)

    return {k: {"score": 7.0, "justification": f"Evaluation fallback: {fallback_msg}"} for k in PILLAR_KEYS}

def weighted_rubric_average(rubric: dict, weights: dict = None) -> float:
    weights = weights or {p["key"]: p["weight"] for p in PILLARS}
    total_w = sum(weights.get(k, 0) for k in rubric if k in weights)
    if total_w == 0:
        return sum(dim["score"] for dim in rubric.values() if isinstance(dim, dict) and "score" in dim) / max(len(rubric), 1)
    return sum(rubric[k]["score"] * weights.get(k, 0) for k in rubric if k in weights and "score" in rubric[k]) / total_w

def generate_actionable_improvements(rubric: dict) -> list[str]:
    improvements = []
    if not rubric:
        return improvements

    sorted_pillars = sorted(
        [(k, v.get("score", 10.0), v.get("justification", "")) for k, v in rubric.items() if isinstance(v, dict)],
        key=lambda x: x[1]
    )

    for key, score, justification in sorted_pillars[:3]:
        if score < 7.5:
            formatted_name = key.replace("_", " ").title()
            improvements.append(f"**Strengthen {formatted_name} (Current Score: {score}/10):** {justification}")

    return improvements

_learned_cache = {"vectorizer": None, "model": None, "version_dir": None, "loaded": False}

def _load_learned_brain():
    if _learned_cache["loaded"]:
        return
    _learned_cache["loaded"] = True

    pointer = os.path.join(IDEA_MODEL_ROOT, "latest.txt")
    if not os.path.exists(pointer):
        return
    with open(pointer) as f:
        version_dir = f.read().strip()

    model_path = os.path.join(version_dir, "model.joblib")
    vec_path = os.path.join(version_dir, "vectorizer.joblib")

    if os.path.exists(vec_path):
        _learned_cache["vectorizer"] = joblib.load(vec_path)
    if os.path.exists(model_path):
        _learned_cache["model"] = joblib.load(model_path)
    _learned_cache["version_dir"] = version_dir

def reload_learned_brain():
    _learned_cache["loaded"] = False
    _load_learned_brain()

def learned_score(idea_text: str):
    _load_learned_brain()
    if _learned_cache["model"] is None or _learned_cache["vectorizer"] is None:
        return None
    X = _learned_cache["vectorizer"].transform([idea_text])
    pred = _learned_cache["model"].predict(X)[0]
    return float(np.clip(pred, 0, 10))

def score_idea(text: str = None, file_bytes: bytes = None, filename: str = None,
               url: str = None, reference_corpus: list[str] = None,
               use_llm: bool = False, api_key: str = None,
               org_context: str = None, pillar_weights: dict = None,
               ollama_base_url: str = None, **kwargs) -> dict:
    idea_text = extract_text(text=text, file_bytes=file_bytes, filename=filename, url=url)
    if len(idea_text.split()) < 5:
        raise ValueError("Extracted text is too short to score.")

    h = heuristic_scores(idea_text, reference_corpus=reference_corpus)
    learned = learned_score(idea_text)

    result = {
        "extracted_text_preview": idea_text[:300],
        "extracted_text_full": idea_text,
        "heuristic": h.__dict__,
        "learned_score": learned,
        "llm_rubric": None,
        "how_to_improve": [],
        "overall_score": h.heuristic_overall,
    }

    weight_values = [(h.heuristic_overall, 0.4 if learned is not None else 1.0)]
    if learned is not None:
        weight_values.append((learned, 0.6))

    if use_llm:
        rubric = llm_rubric_score(idea_text, api_key=api_key, org_context=org_context, ollama_base_url=ollama_base_url)
        result["llm_rubric"] = rubric
        result["how_to_improve"] = generate_actionable_improvements(rubric)
        llm_avg = weighted_rubric_average(rubric, weights=pillar_weights)

        if learned is not None:
            weight_values = [(h.heuristic_overall, 0.15), (learned, 0.35), (llm_avg, 0.50)]
        else:
            weight_values = [(h.heuristic_overall, 0.15), (llm_avg, 0.85)]

    total_weight = sum(w for _, w in weight_values)
    result["overall_score"] = round(sum(v * w for v, w in weight_values) / total_weight, 2)

    return result
