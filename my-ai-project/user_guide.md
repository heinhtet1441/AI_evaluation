# AI Evaluation Model — User Guide

This guide covers three things, in order: **setup**, **training the model**, and **connecting it to a dashboard**. Follow the steps in order the first time — after that, you'll mostly only repeat the "Training" section when you have new data.

## System Overview

```
  [Your Data] → ai_evaluation_pipeline.py → model_artifacts/<version>/
                                                     │
                                                     ▼
                                          serve_api.py (FastAPI)
                                                     │
                                                     ▼
                                          dashboard.py (Streamlit)
```

Three files, three jobs:
- **`ai_evaluation_pipeline.py`** — trains the model and saves it, its scaler, its metrics, and its feature list to a versioned folder.
- **`serve_api.py`** — loads the latest trained model and exposes it over HTTP (`/predict`, `/metrics`).
- **`dashboard.py`** — a web page that calls the API to show metrics and let people run predictions. It never touches the model files directly.
- **`idea_scoring.py`** — extracts text from typed input, an uploaded file, or a URL, then scores the idea it contains (see Part 4).
- **`train_idea_brain.py`** — trains a model on human reviewer feedback so idea scoring improves over time (see Part 5).
- **`data_store.py`** — persistent SQLite storage for every idea submission and every reviewer rating (see Part 6).

See also: **Part 7 — Comparing Multiple Ideas** and **Part 8 — Training From the Dashboard (No Code)**.

---

## Part 1 — Setup (one time)

### Step 1: Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow joblib shap fastapi "uvicorn[standard]" streamlit requests pdfplumber python-docx beautifulsoup4 anthropic plotly --break-system-packages
```

### Step 2: Put all the files in one project folder

```
my-ai-project/
├── ai_evaluation_pipeline.py
├── serve_api.py
├── dashboard.py
├── idea_scoring.py
├── train_idea_brain.py
└── data_store.py
```

### Step 3: Plug in your real data

Open `ai_evaluation_pipeline.py` and find the `load_data()` function near the top. Replace the random-data placeholder with your actual source, for example:

```python
def load_data() -> pd.DataFrame:
    return pd.read_csv("your_company_data.csv")
    # or: pd.read_sql(query, connection)
```

Make sure your dataframe has a column named `Target` (or rename `target_col` in `split_data()` to match your column).

---

## Part 2 — How to Train

### Step 1: Run the training pipeline

```bash
python ai_evaluation_pipeline.py
```

### Step 2: Watch the log output

You'll see lines like:

```
INFO | Data validated: 100 rows, 6 columns
INFO | Training finished after 23 epochs (best weights restored)
INFO | Test MAE: 0.2451
INFO | Test RMSE: 0.2981
INFO | Test R2: 0.0312
INFO | Saved model + scaler to model_artifacts/20260812_101530
```

- **R² close to 1** → the model explains most of the variance (good).
- **R² close to 0 or negative** → the model isn't learning meaningful patterns — usually means you need more data, better features, or more training epochs.

### Step 3: Confirm the artifacts were saved

```bash
ls model_artifacts/
```

You should see a timestamped folder containing `model.keras`, `scaler.joblib`, `metrics.json`, and `feature_names.json`, plus a `latest.txt` pointing to it.

### Step 4: Retrain whenever you have new data

Just re-run `python ai_evaluation_pipeline.py`. Every run creates a **new** version folder and updates `latest.txt` — old versions are never overwritten, so you can always roll back.

---

## Part 3 — How to Integrate with the Dashboard

### Step 1: Start the model API

```bash
uvicorn serve_api:app --port 8000
```

Check it's alive:

```bash
curl http://localhost:8000/health
```

You should get back `{"status": "ok", "model_version": "model_artifacts/2026..."}`.

### Step 2: Start the dashboard (in a second terminal)

```bash
streamlit run dashboard.py
```

This opens a browser tab at `http://localhost:8501` showing:
- Live MAE / MSE / RMSE / R² for the currently deployed model
- A form to test single predictions
- A CSV upload for batch predictions

### Step 3: After retraining, refresh the API without restarting it

```bash
curl -X POST http://localhost:8000/reload-model
```

This tells the running API to pick up the newest `model_artifacts/` version — the dashboard will show the updated metrics on next refresh.

### Step 4 (optional): Embedding into an existing company dashboard

If you already have a BI tool (Power BI, Grafana, an internal web app) instead of Streamlit, you don't need `dashboard.py` at all — just point that tool at the same API:

| Need | Call |
|---|---|
| Latest accuracy numbers | `GET http://localhost:8000/metrics` |
| Run a prediction | `POST http://localhost:8000/predict` with `{"rows": [{...}]}` |
| Which features the model expects | `GET http://localhost:8000/model-info` |

Any tool that can make an HTTP request (Power BI's Web connector, a Grafana JSON datasource plugin, a custom internal page) can consume these endpoints directly.

---

## Day-to-Day Usage Summary

| I want to... | Command |
|---|---|
| Train / retrain the model | `python ai_evaluation_pipeline.py` |
| Start serving predictions | `uvicorn serve_api:app --port 8000` |
| Open the dashboard | `streamlit run dashboard.py` |
| Push a new model live without downtime | `curl -X POST http://localhost:8000/reload-model` |
| Check current model accuracy | `curl http://localhost:8000/metrics` |

---

## Part 4 — How Idea Scoring Works & How Users Use It

Separate from the numeric regression model above, `idea_scoring.py` handles the actual use case: a person submits an **idea** (not a row of numbers), and the system measures it.

### How the measurement works

```
 Source (text / file / URL)
        │
        ▼
 extract_text()        <- pulls plain text out of whatever was submitted
        │                  - typed text: used as-is
        │                  - .pdf/.docx/.txt upload: parsed to text
        │                  - URL: page is fetched and the article body is extracted
        ▼
 heuristic_scores()    <- instant, deterministic, no external API needed
        │                  - Clarity: penalizes run-on, unfocused writing
        │                  - Specificity: rewards concrete detail (numbers, named things)
        │                  - Length adequacy: too short = thin idea, too long = unfocused
        │                  - Novelty: TF-IDF similarity vs. every idea submitted before —
        │                    the more it overlaps with past submissions, the lower it scores
        ▼
 llm_rubric_score()    <- optional, deeper qualitative judgement against the
                           14-Dimension Idea Evaluation Rubric (4 general +
                           10-pillar breakdown; needs an
                           Anthropic API key set as ANTHROPIC_API_KEY on the
                           server). Each pillar is scored 1-10 with a
                           justification.
        ▼
 overall_score          <- heuristic-only if LLM judgement is off,
                            or a blend of heuristic + LLM (weighted rubric
                            average) + trained brain if available
```

Each score comes with a **reason**, not just a number — that's the explainability pillar: nobody should get a "3/10" with no idea why.

### The 14-Dimension Idea Evaluation Rubric

When "deeper AI judgement" is turned on, the LLM scores every idea against all fourteen dimensions, each 1-10 with a justification — the original 4 general dimensions, plus the expanded 10-pillar breakdown:

| # | Dimension | What it measures |
|---|---|---|
| 1 | Feasibility (general) | Quick overall read on whether it's realistically buildable |
| 2 | Market Potential (general) | Quick overall read on whether there's a meaningful audience/opportunity |
| 3 | Innovation (general) | Quick overall read on how creative or original it feels |
| 4 | Risk (general) | Quick overall read on downside risk (higher score = lower risk) |
| 5 | Novelty & Differentiation | How unique the idea is vs. existing solutions, research, patents, or prior art |
| 6 | Technical & Operational Feasibility | Whether it can actually be built today with current technology and infrastructure |
| 7 | Value Creation & Market Impact | The size of the payoff if it succeeds — revenue, cost savings, time saved, quality of life |
| 8 | Problem Definition & Groundedness | Whether the underlying problem statement is logically sound and evidence-based |
| 9 | Scalability & Growth Potential | Whether unit economics hold up (or break down) at 10x–100x scale |
| 10 | Risk, Safety & Compliance | Legal, regulatory, safety, security, and ethical exposure (higher score = lower risk) |
| 11 | Resource & Cost Efficiency | Whether the expected ROI justifies the engineering time, budget, and hardware needed |
| 12 | Adoption Friction & User Usability | How much users have to change their behavior to adopt it (higher score = lower friction) |
| 13 | Environmental & Societal Sustainability | Carbon footprint, labor displacement, and societal equity impact (higher score = more sustainable) |
| 14 | Strategic & Goal Alignment | Fit with your specific organization's mission, OKRs, and roadmap |

The first four are a fast, general gut-check; the remaining ten give the same territory a much more detailed, structured look (e.g. dimension 1's "Feasibility" vs. dimension 6's "Technical & Operational Feasibility" — the general read plus the deep-dive). Keeping both means you get a quick signal *and* the detailed breakdown in one pass.

For dimension 14 to be meaningful, give the dashboard some **organization context** (mission, current OKRs, roadmap focus) in the optional text box that appears when "deeper AI judgement" is checked — without it, the LLM scores general strategic soundness only and flags that org-specific fit couldn't be assessed.

### Weighting the dimensions to your priorities

Not every organization weighs these fourteen equally — a safety-critical company might weight **Risk, Safety & Compliance** heavily; an early-stage startup might weight **Value Creation & Market Impact** and **Novelty** more. Adjust this in `idea_scoring.py`:

```python
# in PILLARS, change the "weight" value for any pillar, e.g.:
{
    "key": "risk_safety_compliance",
    ...
    "weight": 3.0,   # 3x the influence of a default (1.0) pillar
},
```

Or pass custom weights per-request via `pillar_weights` in `score_idea()` / the `/score-idea` API if different teams need different priorities without editing the shared file.

### How a user actually uses it

1. Open the dashboard (`streamlit run dashboard.py`) and go to the **"Submit an Idea"** tab.
2. Pick how they're providing the idea:
   - **Type it in** — write/paste a description directly
   - **Upload a file** — a `.pdf`, `.docx`, or `.txt` write-up
   - **Paste a URL** — a link to a doc, wiki page, or article describing the idea
3. Optionally tick **"Use deeper AI judgement"** if they want the Feasibility/Market Potential/Innovation/Risk breakdown, not just the instant structural score.
4. Click **Score this idea**. Results appear immediately:
   - An overall score out of 10
   - The heuristic breakdown (clarity, specificity, length fit, novelty)
   - The AI judgement breakdown, if enabled
   - A preview of the extracted text, so they can confirm the system read the right content

No coding needed on the user's side — everything above is point-and-click in the browser.

### Turning on the deeper AI judgement

The heuristic scores work immediately with no setup. To enable the LLM rubric scoring:

```bash
export ANTHROPIC_API_KEY="your-key-here"
uvicorn serve_api:app --port 8000
```

Without this set, `use_llm=True` requests will fail with a clear error — heuristic-only scoring still works fine either way.

### Tuning the scoring to your company's standards

Everything is adjustable in `idea_scoring.py`:
- `PILLARS` — the ten rubric pillars' descriptions, focus questions, and weights (see "Weighting the pillars" above)
- The `ideal_min`/`ideal_max` word counts in `heuristic_scores()` — set to whatever length a "properly written" idea looks like at your company
- The weights in `heuristic_overall` (currently clarity 25% / specificity 25% / length 20% / novelty 30%) — rebalance based on what actually matters most for your use case

---

## Part 5 — Training the Idea-Scoring "Brain"

Heuristics and the LLM rubric are useful out of the box, but neither one *learns* — clarity/novelty rules are fixed, and the LLM only knows general good-idea judgement, not your company's specific preferences. `train_idea_brain.py` is what actually trains a model on **your own reviewers' ratings**, so it gets sharper the more feedback it sees.

### Step 1: Collect feedback

Every time someone scores an idea in the dashboard, a **"Reviewer feedback"** slider appears underneath the result. A reviewer enters what they actually think the idea is worth (0-10) and clicks **Submit rating**. This is logged automatically to `idea_feedback.csv` (idea text + score + timestamp) via the `/submit-feedback` API endpoint.

Keep this running as part of normal usage — the more ratings collected, the better the trained model gets.

### Step 2: Train once you have enough ratings

You need at least **20 rated ideas** before training is worthwhile (fewer than that and the model just memorizes noise). Once you do:

```bash
python train_idea_brain.py
```

You'll see output like:

```
INFO | Loaded 47 feedback rows
INFO | MAE: 0.83
INFO | R2: 0.41
INFO | Saved trained idea-scoring brain to idea_model_artifacts/20260812_113000
```

- **R² above ~0.3-0.4** → the model is picking up a real pattern in what your reviewers value.
- **R² near 0 or negative, or a warning about low R²** → not enough signal yet, usually because there isn't enough feedback or reviewers are rating inconsistently. Keep collecting and retrain later — heuristic/LLM scoring keeps working fine in the meantime.

### Step 3: Put the trained brain live

```bash
curl -X POST http://localhost:8000/reload-idea-brain
```

From this point on, every `/score-idea` response includes a `learned_score` — the model's own prediction of what your reviewers would say — and it's automatically blended into `overall_score` alongside the heuristics and LLM judgement.

### Step 4: Retrain on a schedule

Re-run Step 2 + 3 periodically (weekly is reasonable to start) as more feedback accumulates. Each run creates a new versioned folder under `idea_model_artifacts/` — nothing is overwritten, so you can always see how the model's accuracy (MAE/R²) is trending over time by comparing each version's `metrics.json`.

### Why this design

This mirrors the same "continuous monitoring + human-in-the-loop" pattern the numeric model uses in Part 2: automated scoring handles routine cases fast, but the loop stays anchored to real human judgement instead of drifting off on its own — exactly the balance a good AI evaluation system needs.

---

## Part 6 — How Data Is Stored

Everything the system needs to remember lives in two kinds of storage: **versioned files** for trained models, and a **SQLite database** for everything that accumulates over time (submissions, feedback).

### Trained models — files on disk

```
model_artifacts/<timestamp>/       <- the numeric regression model (Part 2)
    model.keras
    scaler.joblib
    feature_names.json
    metrics.json
model_artifacts/latest.txt         <- points to the current version

idea_model_artifacts/<timestamp>/  <- the trained idea-scoring "brain" (Part 5)
    vectorizer.joblib
    model.joblib
    metrics.json
idea_model_artifacts/latest.txt
```

Every training run creates a new timestamped folder — nothing is overwritten, so you can always roll back or compare versions. This is normal practice for ML models: they're binary artifacts, not the kind of thing you'd put in a database row.

### Everything else — `idea_evaluation.db` (SQLite)

A single file, created automatically the first time `serve_api.py` starts. Two tables:

| Table | What it holds | Used by |
|---|---|---|
| `idea_submissions` | Every idea ever scored — full text, heuristic breakdown, LLM rubric, learned score, overall score, timestamp | Novelty checking (comparing new ideas against past ones) and an audit trail of everything submitted |
| `idea_feedback` | Every reviewer rating (idea text + human score + timestamp) | `train_idea_brain.py` — this is exactly what the model trains on |

You can browse what's stored anytime:

```bash
curl http://localhost:8000/submissions?limit=20
```

Or query the database file directly with any SQLite tool:

```bash
sqlite3 idea_evaluation.db "SELECT overall_score, submitted_at FROM idea_submissions ORDER BY id DESC LIMIT 10;"
```

### Backing it up

Because it's one file, backup is just copying it:

```bash
cp idea_evaluation.db backups/idea_evaluation_$(date +%Y%m%d).db
```

Set that up as a daily cron job / scheduled task once this is running for real. Do the same periodically for the `model_artifacts/` and `idea_model_artifacts/` folders — or better, keep the whole project directory under normal file backup / version control, excluding the large `.keras`/`.joblib` binaries if storage is a concern.

### Scaling beyond SQLite

SQLite comfortably handles a single server with thousands to low-millions of rows — plenty for most companies' idea volumes. If you outgrow it (multiple servers writing at once, need for a shared company data warehouse), `data_store.py` is the only file to change: swap the SQLite connection for Postgres via SQLAlchemy/psycopg2, keeping the same function names (`save_submission`, `get_reference_corpus`, `save_feedback`, `get_feedback_df`) so nothing else in the project needs to change.

---

---

## Part 7 — Comparing Multiple Ideas

The **"Compare Ideas"** tab in the dashboard scores 2–6 ideas at once and shows which one wins, by how much, and on which specific dimensions — not just a single number each.

### How to use it

1. Open the dashboard and go to **Compare Ideas**.
2. Choose how you're providing the ideas:
   - **Manual** — pick 2–6 slots, and for each one independently choose Text / File / URL (mix and match freely — idea 1 could be pasted text, idea 2 a PDF, idea 3 a URL).
   - **Upload a CSV of ideas** — one file with an `idea_text` column (and optional `label` column) fills in up to 6 rows automatically. Useful when you've already collected several idea write-ups in a spreadsheet.
3. Optionally tick **"Use deeper AI judgement"** to score every idea against the full 14-dimension rubric — this is what unlocks the full comparison matrix below (without it, you still get the heuristic comparison, just not the rubric rows).
4. Click **Compare these ideas**.

### What you get back

- **A headline result** — which idea scored highest, and by how many points versus each other idea, right at the top so it doesn't require reading a table.
- **A ranking bar chart** — every idea's overall score, sorted, with the leader highlighted in green.
- **A radar chart** — clarity / specificity / length fit / novelty overlaid for every idea at once, so you can see *shape* differences (e.g. one idea might win on novelty but lose on clarity) that a single overall number would hide.
- **The measurement matrix** — every scored dimension as a row, every idea as a column, with the best-scoring idea highlighted per row. If deeper AI judgement was on, this includes all 14 rubric dimensions plus the learned score (if a brain has been trained) and heuristics, all in one table. This is the "which idea is better on what, specifically" view.

---

## Part 8 — Training From the Dashboard (No Code)

Staff doing research shouldn't need to open a terminal or touch `idea_scoring.py`/`train_idea_brain.py` to add data or retrain. The **"Train the Brain"** tab in the dashboard does everything Part 5 covered, but entirely through file uploads and buttons.

### 1. Quick CSV import

If research has already been organized into a spreadsheet with columns `idea_text` and `human_score`, upload it directly — it's previewed, then one button ("Add this CSV to the training set") imports every row.

### 2. Upload raw research documents

For research that's still in raw form — PDFs, Word docs, plain text files gathered from wherever — upload as many as needed at once. Each file's text is extracted automatically (no need to open the files or copy-paste), and an editable table appears with a `score` column defaulting to 5.0. Staff review each extracted preview and adjust the score to reflect their actual assessment, then click **"Add these documents to the training set."**

### 3. Train

Once enough data is in (the tab shows a live count, and flags when you're under the 20-row minimum from Part 5), click **Train now**. This runs the exact same training process as `python train_idea_brain.py` — but from a button, and it **automatically puts the new model live** afterward (no separate reload step needed). MAE/RMSE/R² are shown immediately so staff can see whether the new training run actually improved things.

Nothing in this flow touches source code — uploading, scoring, and retraining are all done from the browser.

---

## Troubleshooting

- **"No trained model found" when starting the API** → you haven't run the training pipeline yet, or you're running `uvicorn` from a different folder than `model_artifacts/`. Run training first, and start the API from the same directory.
- **Dashboard shows "Cannot reach the model API"** → the API isn't running, or it's on a different port than `API_URL` in `dashboard.py`. Update `API_URL` if you changed the port.
- **R² is very low / negative** → this is a data problem, not a code problem — check that `load_data()` is returning your real, cleaned data and not the random placeholder.
- **`shap` import errors during training** → explainability is optional; the pipeline logs a warning and continues without it. Install with `pip install shap --break-system-packages` if you want feature-importance output.
