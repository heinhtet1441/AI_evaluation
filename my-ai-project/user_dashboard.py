import json
import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

# 1. Cloud & Local dynamic configuration (Valid HTTPS URL fallback)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://ai-evaluation-8ju7.onrender.com").rstrip("/")
NGROK_OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "https://dexterous-nanny-amniotic.ngrok-free.dev/v1")

# Ensure /v1 path exists for OpenAI client format
if not NGROK_OLLAMA_URL.endswith("/v1"):
    OLLAMA_OPENAI_URL = f"{NGROK_OLLAMA_URL.rstrip('/')}/v1"
else:
    OLLAMA_OPENAI_URL = NGROK_OLLAMA_URL

# OpenAI Client for Ollama
ollama_client = OpenAI(
    base_url=OLLAMA_OPENAI_URL,
    api_key="ollama",
    default_headers={"Bypass-Tunnel-Remainder": "true"}  # LocalTunnel support
)

# Configure Page Layout
st.set_page_config(
    page_title="AI Evaluator Enterprise Pro",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Fully Mobile + Tablet Responsive Modern Dark Theme CSS Architecture
MOBILE_RESPONSIVE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;700;800&display=swap');

    :root {
        --gap-2: 0.5rem;
        --gap-3: 0.75rem;
        --gap-4: 1rem;
        --gap-6: 1.5rem;
    }

    html, body {
        overflow-x: hidden !important;
    }

    /* Global Glow Reset */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #030014 !important;
        color: #f1f5f9 !important;
        -webkit-tap-highlight-color: transparent;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, #2e0854, #030014 75%) !important;
        width: 100% !important;
        max-width: min(100vw - 16px, 460px) !important;
        margin: 8px auto !important;
        border: 1px solid rgba(236, 72, 153, 0.3) !important;
        border-radius: 28px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.95), 0 0 35px rgba(168, 85, 247, 0.2) !important;
        transition: max-width 0.2s ease;
    }

    @media (min-width: 640px) {
        .stApp {
            max-width: min(100vw - 48px, 680px) !important;
            margin: 20px auto !important;
        }
    }

    @media (min-width: 1024px) {
        .stApp {
            max-width: 820px !important;
            margin: 32px auto !important;
        }
    }

    header[data-testid="stHeader"], footer,
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        visibility: hidden !important;
        height: 0px !important;
        display: none !important;
    }

    div[data-testid="stVerticalBlock"] {
        gap: var(--gap-3) !important;
    }

    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: var(--gap-2) !important;
    }

    div[data-testid="stHorizontalBlock"] > div {
        min-width: 0 !important;
    }

    .main .block-container {
        padding: var(--gap-3) 12px var(--gap-4) 12px !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    @media (min-width: 640px) {
        .main .block-container {
            padding: var(--gap-4) 22px var(--gap-6) 22px !important;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: var(--gap-3) !important;
        }
    }

    @media (min-width: 1024px) {
        .main .block-container {
            padding: var(--gap-6) 30px var(--gap-6) 30px !important;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: var(--gap-4) !important;
        }
    }

    .app-top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 14px;
        background: rgba(15, 5, 29, 0.85);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(236, 72, 153, 0.3);
        border-top-left-radius: 26px;
        border-top-right-radius: 26px;
        margin: calc(-1 * var(--gap-3)) -12px var(--gap-3) -12px;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.25);
    }

    @media (min-width: 640px) {
        .app-top-header {
            padding: 16px 22px;
            margin: calc(-1 * var(--gap-4)) -22px var(--gap-4) -22px;
        }
    }

    @media (min-width: 1024px) {
        .app-top-header {
            padding: 20px 30px;
            margin: calc(-1 * var(--gap-6)) -30px var(--gap-6) -30px;
        }
    }

    .brand-group {
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 800;
        font-size: 1rem;
        background: linear-gradient(90deg, #f43f5e, #a855f7, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    @media (min-width: 640px) {
        .brand-group { font-size: 1.2rem; }
    }

    .avatar-pill {
        width: 30px;
        height: 30px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        color: #bec6e0;
    }

    .glass-panel {
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }

    @media (min-width: 640px) {
        .glass-panel { padding: 18px 20px; }
    }

    .status-row-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 16px;
        padding: 10px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.15);
    }

    .status-title-text {
        font-size: 0.6rem;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .status-online-pill {
        font-size: 0.75rem;
        font-weight: 600;
        color: #10b981;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 2px;
    }

    .status-dot-green {
        width: 7px;
        height: 7px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #10b981;
    }

    .section-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 0;
    }

    .section-heading {
        font-size: 0.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }

    @media (min-width: 640px) {
        .section-heading { font-size: 1rem; }
    }

    .score-hero-box {
        text-align: center;
        padding: 18px 12px;
        background: radial-gradient(circle at center, rgba(168, 85, 247, 0.25) 0%, rgba(15, 23, 42, 0.9) 100%);
        border-radius: 20px;
        border: 1px solid rgba(168, 85, 247, 0.4);
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.2);
    }

    @media (min-width: 640px) {
        .score-hero-box { padding: 28px 20px; }
    }

    .badge-recommended {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        font-weight: 800;
        font-size: 0.65rem;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-revisions {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        font-weight: 800;
        font-size: 0.65rem;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.4);
        font-family: 'JetBrains Mono', monospace;
    }

    .badge-danger {
        background: rgba(243, 24, 104, 0.2);
        color: #f31868;
        border: 1px solid #f31868;
        font-weight: 800;
        font-size: 0.65rem;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        letter-spacing: 0.5px;
        box-shadow: 0 0 12px rgba(243, 24, 104, 0.4);
        font-family: 'JetBrains Mono', monospace;
    }

    .score-num-text {
        font-size: clamp(3.4rem, 8vw, 5rem);
        font-weight: 900;
        background: linear-gradient(180deg, #ffffff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0 2px 0;
        line-height: 1;
    }

    .compare-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        font-size: 0.75rem;
        color: #94a3b8;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .compare-metric-row:last-child {
        border-bottom: none;
    }

    div.stButton > button {
        width: 100% !important;
        min-height: 44px !important;
        background: linear-gradient(135deg, #ec4899, #8b5cf6, #06b6d4) !important;
        background-size: 200% 200% !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-size: 0.82rem !important;
        padding: 10px 14px !important;
        box-shadow: 0 0 18px rgba(236, 72, 153, 0.35) !important;
        animation: gradientMove 6s ease infinite;
        transition: all 0.2s ease !important;
    }

    @media (min-width: 640px) {
        div.stButton > button {
            font-size: 0.92rem !important;
            padding: 12px 18px !important;
        }
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    div.stButton > button:active {
        transform: scale(0.98) !important;
    }

    div.stDownloadButton > button {
        width: 100% !important;
        min-height: 42px !important;
        background: rgba(30, 41, 59, 0.8) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.72rem !important;
        padding: 8px !important;
    }

    @media (min-width: 640px) {
        div.stDownloadButton > button { font-size: 0.8rem !important; }
    }

    div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
        background-color: rgba(15, 23, 42, 0.85) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        border-radius: 10px !important;
        font-size: 0.8rem !important;
    }

    @media (min-width: 640px) {
        div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
            font-size: 0.92rem !important;
        }
    }

    div[data-testid="stMarkdownContainer"] p {
        font-size: 0.78rem !important;
    }

    @media (min-width: 640px) {
        div[data-testid="stMarkdownContainer"] p { font-size: 0.88rem !important; }
    }

    .st-key-back_btn_wrap div.stButton > button {
        width: auto !important;
        min-height: 36px !important;
        background: rgba(30, 41, 59, 0.9) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        color: #e9d5ff !important;
        border-radius: 10px !important;
        padding: 4px 10px !important;
        font-size: 0.72rem !important;
        box-shadow: none !important;
    }

    .st-key-bottom_nav_wrap {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 999 !important;
        background: rgba(15, 5, 29, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(236, 72, 153, 0.3) !important;
        border-radius: 18px !important;
        padding: 4px 2px !important;
        margin-top: var(--gap-4) !important;
        box-shadow: 0 -5px 25px rgba(0, 0, 0, 0.6);
    }

    @media (min-width: 640px) {
        .st-key-bottom_nav_wrap {
            max-width: 420px;
            margin-left: auto !important;
            margin-right: auto !important;
            padding: 6px 8px !important;
        }
    }

    .st-key-bottom_nav_wrap div[data-testid="stHorizontalBlock"] {
        justify-content: space-between !important;
    }

    .st-key-bottom_nav_wrap div.stButton > button {
        width: 100% !important;
        min-height: 38px !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #94a3b8 !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        padding: 4px 2px !important;
        border-radius: 8px !important;
        animation: none !important;
    }

    .st-key-bottom_nav_wrap button[kind="primary"] {
        color: #a855f7 !important;
        background: rgba(168, 85, 247, 0.18) !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        font-weight: 800 !important;
        box-shadow: 0 0 10px rgba(168, 85, 247, 0.3) !important;
    }
</style>
"""
st.markdown(MOBILE_RESPONSIVE_CSS, unsafe_allow_html=True)

# State Management
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Dashboard"

if "nav_history" not in st.session_state:
    st.session_state["nav_history"] = ["Dashboard"]


def navigate_to(screen_name):
    if st.session_state["active_nav"] != screen_name:
        st.session_state["nav_history"].append(screen_name)
        st.session_state["active_nav"] = screen_name


def go_back():
    if len(st.session_state.get("nav_history", [])) > 1:
        st.session_state["nav_history"].pop()
        st.session_state["active_nav"] = st.session_state["nav_history"][-1]
    else:
        st.session_state["active_nav"] = "Dashboard"


def score_tier(overall: float):
    if overall >= 8.0:
        return "badge-recommended", "● HIGHLY RECOMMENDED", \
            "Strong structural alignment with strategic metrics; the proposal demonstrates high clarity and robust commercial viability."
    elif overall >= 6.0:
        return "badge-revisions", "● CONSIDER WITH REVISIONS", \
            "The proposal shows promise but has notable gaps in clarity, specificity, or feasibility worth addressing before moving forward."
    else:
        return "badge-danger", "● HIGH RISK / REJECT", \
            "The proposal has significant weaknesses in strategic alignment, feasibility, or clarity that pose a high risk to successful execution."


try:
    health_resp = requests.get(f"{API_BASE_URL}/health", timeout=15)
    api_online = health_resp.status_code == 200
    model_ver = health_resp.json().get("model_version", "Active") if api_online else "Offline"
except Exception:
    api_online = False
    model_ver = "Offline"

# App Top Header
if st.session_state["active_nav"] != "Dashboard":
    with st.container(key="back_btn_wrap"):
        col_back, col_head = st.columns([1, 4])
        with col_back:
            if st.button("← Back", key="btn_back_key"):
                go_back()
                st.rerun()
        with col_head:
            st.markdown("""<div class="app-top-header" style="border-top-left-radius:0;"><div class="brand-group"><span>AI Evaluator Pro</span></div><div class="avatar-pill">👤</div></div>""", unsafe_allow_html=True)
else:
    st.markdown("""<div class="app-top-header"><div class="brand-group"><span>AI Evaluator Pro</span></div><div style="display:flex; align-items:center; gap:8px;"><span style="font-size:1rem; color:#94a3b8;">🔔</span><div class="avatar-pill" style="border: 2px solid #a855f7;">👤</div></div></div>""", unsafe_allow_html=True)

# ===========================================================================
# SCREEN 1: DASHBOARD
# ===========================================================================
if st.session_state["active_nav"] == "Dashboard":
    st.markdown(f"""<div class="status-row-card"><div><div class="status-title-text">SYSTEM STATUS</div><div class="status-online-pill"><span class="status-dot-green"></span>{'API Online' if api_online else 'API Offline'}</div></div><div style="text-align: right;"><div class="status-title-text">ENGINE</div><div style="font-size:0.75rem; font-weight:600; color:#d4e4fa; margin-top:2px;">Model: {model_ver}</div></div></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="glass-panel"><div style="font-size:1.3rem; margin-bottom:4px;">📄</div><h3 style="margin:0 0 4px 0; color:#f8fafc; font-size:1rem; font-weight:700;">Start New Evaluation</h3><p style="margin:0; color:#94a3b8; font-size:0.78rem; line-height:1.4;">Upload proposal files or paste texts for deep learning multidimensional evaluation.</p></div>""", unsafe_allow_html=True)

    if st.button("🚀 New Evaluation", key="dash_btn_new_eval"):
        navigate_to("Evaluate")
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 10px; text-align: center;"><div style="font-size:1.1rem; margin-bottom:2px;">⚔️</div><div style="font-weight:700; font-size:0.75rem; color:#f8fafc;">Compare</div></div>""", unsafe_allow_html=True)
        if st.button("Open Compare", key="dash_btn_sub_comp", use_container_width=True):
            navigate_to("Compare")
            st.rerun()
    with c2:
        st.markdown("""<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 10px; text-align: center;"><div style="font-size:1.1rem; margin-bottom:2px;">📁</div><div style="font-weight:700; font-size:0.75rem; color:#f8fafc;">Saved Reports</div></div>""", unsafe_allow_html=True)
        if st.button("View Reports", key="dash_btn_sub_reports", use_container_width=True):
            navigate_to("Evaluate")
            st.rerun()

    st.markdown("<h4 class='section-heading' style='margin-top: var(--gap-2);'>Recent Activity</h4>", unsafe_allow_html=True)

    st.markdown("""<div class="glass-panel" style="display:flex; justify-content:space-between; align-items:center; padding:10px 12px;"><div style="display:flex; align-items:center; gap:8px;"><span style="font-size:1rem;">📄</span><div><div style="font-weight:700; font-size:0.78rem; color:#f8fafc;">Project Helios</div><div style="font-size:0.68rem; color:#94a3b8;">2 hours ago</div></div></div><div style="font-size:0.95rem; font-weight:800; color:#10b981;">8.4 / 10</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="glass-panel" style="display:flex; justify-content:space-between; align-items:center; padding:10px 12px;"><div style="display:flex; align-items:center; gap:8px;"><span style="font-size:1rem;">📄</span><div><div style="font-weight:700; font-size:0.78rem; color:#f8fafc;">Atlas Migration</div><div style="font-size:0.68rem; color:#94a3b8;">Yesterday</div></div></div><div style="font-size:0.95rem; font-weight:800; color:#f59e0b;">6.2 / 10</div></div>""", unsafe_allow_html=True)

# ===========================================================================
# SCREEN 2: EVALUATE
# ===========================================================================
elif st.session_state["active_nav"] == "Evaluate":
    st.markdown("<h2 style='font-size:1.1rem; font-weight:800; color:#f8fafc; margin:0;'>New Evaluation</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin:0;'>Submit data for multidimensional AI analysis.</p>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Format",
        ["Manual Entry", "Document Upload", "CSV Upload"],
        horizontal=True,
        key="eval_mode_radio"
    )

    idea_text_payload = None
    file_payload = None

    if input_mode == "Manual Entry":
        idea_text_input = st.text_area(
            "Proposal Text",
            height=120,
            placeholder="Paste initiative details here...",
            value="Enterprise Cybersecurity SOC Automation platform featuring eBPF kernel-level event tracing and graph neural networks..."
        )
        if idea_text_input.strip():
            idea_text_payload = idea_text_input.strip()

    elif input_mode == "Document Upload":
        uploaded_doc = st.file_uploader("Upload Proposal (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
        if uploaded_doc:
            file_payload = uploaded_doc

    elif input_mode == "CSV Upload":
        uploaded_csv = st.file_uploader("Upload CSV ('idea_text')", type=["csv"])
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                if df.empty:
                    st.error("This CSV has no rows.")
                elif "idea_text" in df.columns:
                    idea_text_payload = str(df["idea_text"].iloc[0])
                else:
                    st.error("This CSV has no 'idea_text' column.")
            except Exception as e:
                st.error(f"Couldn't read that CSV: {e}")

    org_context = st.text_input("Organization Context (Optional)", placeholder="e.g., Q3 OKRs...", value="Q3 OKRs: Enterprise Reliability")
    use_llm = st.toggle("Deep 10-Pillar Rubric Matrix", value=True)

    eval_triggered = st.button("🪄 Evaluate Idea", key="btn_trigger_evaluation")

    if eval_triggered:
        if not file_payload and not idea_text_payload:
            st.warning("Please enter some proposal text or upload a file before evaluating.")
        else:
            st.session_state.pop("result", None)
            st.session_state.pop("comp_results", None)
            st.session_state.pop("chat_history", None)
            with st.spinner("Analyzing proposal across dimensions..."):
                try:
                    headers = {"Bypass-Tunnel-Remainder": "true"}
                    if file_payload:
                        files = {"file": (file_payload.name, file_payload.getvalue(), file_payload.type)}
                        data = {"use_llm": str(use_llm).lower(), "org_context": org_context}
                        res = requests.post(f"{API_BASE_URL}/score-idea", files=files, data=data, headers=headers, timeout=60)
                    else:
                        payload = {"text": idea_text_payload, "use_llm": str(use_llm).lower(), "org_context": org_context}
                        res = requests.post(f"{API_BASE_URL}/score-idea", data=payload, headers=headers, timeout=60)

                    if res.status_code == 200:
                        st.session_state["result"] = res.json()
                        st.success("Evaluation complete! View graphs below.")
                    else:
                        st.error(f"Server Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection Error ({API_BASE_URL}): {e}")

    if "result" in st.session_state:
        res = st.session_state["result"]
        overall = res.get("overall_score", 0.0)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        badge_class, badge_label, hero_desc = score_tier(overall)
        st.markdown(f"""<div class="score-hero-box"><div class="{badge_class}">{badge_label}</div><div class="score-num-text">{overall}<span style="font-size:1.1rem; color:#94a3b8;"> / 10.0</span></div><p style="font-size:0.75rem; color:#94a3b8; margin:4px 0 0 0;">{hero_desc}</p></div>""", unsafe_allow_html=True)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button("📥 Report (JSON)", json.dumps(res, indent=2), "report.json", "application/json", use_container_width=True)
        with col_exp2:
            flat_summary = {"overall_score": overall, **res.get("heuristic", {})}
            st.download_button("📊 Summary (CSV)", pd.DataFrame([flat_summary]).to_csv(index=False), "summary.csv", "text/csv", use_container_width=True)

        # Bar Chart
        rubric = res.get("llm_rubric")
        if rubric:
            st.markdown("<h4 class='section-heading' style='margin-top: var(--gap-2);'>📊 Strategic Pillar Scores Chart</h4>", unsafe_allow_html=True)

            cats = [k.replace('_', ' ').title() for k in rubric.keys()]
            scs = [v.get("score", 5.0) if isinstance(v, dict) else 5.0 for v in rubric.values()]

            chart_df = pd.DataFrame({"Pillar": cats, "Score": scs})
            chart_height = max(180, 42 * len(cats) + 50)

            fig = px.bar(
                chart_df,
                x="Score",
                y="Pillar",
                orientation='h',
                color="Score",
                color_continuous_scale=["#f31868", "#f59e0b", "#10b981"],
                text="Score"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0, 10.5], color='#94a3b8', gridcolor='rgba(255,255,255,0.05)', title=""),
                yaxis=dict(color='#f8fafc', autorange="reversed", title=""),
                margin=dict(l=0, r=15, t=5, b=5),
                height=chart_height,
                coloraxis_showscale=False
            )
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True, key="single_eval_chart")

            st.markdown("<h4 class='section-heading' style='margin-top: var(--gap-2);'>🎛️ Strategic Pillars Breakdown</h4>", unsafe_allow_html=True)
            for k, v in rubric.items():
                s_val = v.get("score", 5.0) if isinstance(v, dict) else 5.0
                j_val = v.get("justification", "") if isinstance(v, dict) else str(v)
                formatted_k = k.replace('_', ' ').title()

                clr = "#10b981" if s_val >= 7.5 else ("#f59e0b" if s_val >= 5.5 else "#f31868")
                st.markdown(f"**{formatted_k}** — <span style='color:{clr}; font-weight:800;'>{s_val}/10</span>", unsafe_allow_html=True)
                st.caption(j_val)
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ===========================================================================
# SCREEN 3: COMPARE
# ===========================================================================
elif st.session_state["active_nav"] == "Compare":
    st.markdown("<h2 style='font-size:1.1rem; font-weight:800; color:#f8fafc; margin:0;'>Compare Proposals</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin:0;'>Benchmark candidates side-by-side with multidimensional metrics.</p>", unsafe_allow_html=True)

    use_llm_comp = st.toggle("Enable Deep 10-Pillar Rubric", value=True, key="comp_llm_toggle")
    num_ideas = st.slider("Number of Candidates (2 to 4)", min_value=2, max_value=4, value=2, key="comp_num_slider")
    comp_org_context = st.text_input("Organization Context (Optional)", key="comp_org_context")

    candidates_payload = []

    for i in range(num_ideas):
        accent_clr = "#ec4899" if i == 0 else ("#06b6d4" if i == 1 else ("#a855f7" if i == 2 else "#10b981"))
        st.markdown(f"""<div class="glass-panel" style="padding:8px 10px; border-left: 3px solid {accent_clr};"><div style="font-size:0.72rem; font-weight:700; color:{accent_clr}; font-family:'JetBrains Mono', monospace; text-transform:uppercase;">Candidate {i+1}</div></div>""", unsafe_allow_html=True)

        c_label = st.text_input(f"Label {i+1}", value=f"Proposal {chr(65+i)}", key=f"comp_lbl_{i}")
        c_mode = st.radio(f"Input Format {i+1}", ["Manual Entry", "Document Upload", "CSV Upload"], horizontal=True, key=f"comp_mode_{i}")

        c_text, c_file = None, None
        if c_mode == "Manual Entry":
            c_text = st.text_area(f"Content {i+1}", height=80, key=f"comp_txt_{i}")
        elif c_mode == "Document Upload":
            c_file = st.file_uploader(f"Upload Doc {i+1}", type=["pdf", "docx", "txt"], key=f"comp_doc_{i}")
        elif c_mode == "CSV Upload":
            c_csv = st.file_uploader(f"Upload CSV {i+1}", type=["csv"], key=f"comp_csv_{i}")
            if c_csv:
                try:
                    df_c = pd.read_csv(c_csv)
                    if "idea_text" in df_c.columns and not df_c.empty:
                        c_text = str(df_c["idea_text"].iloc[0])
                    else:
                        st.error(f"Candidate {i+1}: CSV has no 'idea_text' column or no rows.")
                except Exception as e:
                    st.error(f"Candidate {i+1}: couldn't read that CSV ({e}).")

        candidates_payload.append({"label": c_label, "text": c_text, "file": c_file})

    if st.button("⚔️ Execute Benchmark Comparison", key="btn_compare_exec"):
        st.session_state.pop("result", None)
        st.session_state.pop("comp_results", None)
        st.session_state.pop("chat_history", None)

        results = []
        errors = []
        with st.spinner("Benchmarking candidates side-by-side..."):
            for cand in candidates_payload:
                try:
                    res = None
                    headers = {"Bypass-Tunnel-Remainder": "true"}
                    if cand["file"]:
                        files = {"file": (cand["file"].name, cand["file"].getvalue(), cand["file"].type)}
                        res = requests.post(
                            f"{API_BASE_URL}/score-idea", files=files,
                            data={"use_llm": str(use_llm_comp).lower(), "org_context": comp_org_context},
                            headers=headers,
                            timeout=60,
                        )
                    elif cand["text"] and cand["text"].strip():
                        res = requests.post(
                            f"{API_BASE_URL}/score-idea",
                            data={"text": cand["text"], "use_llm": str(use_llm_comp).lower(), "org_context": comp_org_context},
                            headers=headers,
                            timeout=60,
                        )

                    if res is None:
                        continue
                    if res.status_code == 200:
                        d = res.json()
                        d["label"] = cand["label"]
                        results.append(d)
                    else:
                        errors.append(f"{cand['label']}: server error ({res.status_code})")
                except Exception as e:
                    errors.append(f"{cand['label']}: {e}")

        if errors:
            st.error("Some candidates failed to score:\n" + "\n".join(errors))

        if len(results) >= 2:
            st.session_state["comp_results"] = results
            st.rerun()
        elif not errors:
            st.error("Please provide valid content or files for at least 2 candidates.")

    if "comp_results" in st.session_state:
        comp_data = st.session_state["comp_results"]
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("<h4 class='section-heading' style='margin-top: var(--gap-2);'>📋 Benchmarking Matrix</h4>", unsafe_allow_html=True)

        res_cols = st.columns(len(comp_data))
        for idx, item in enumerate(comp_data):
            accent_color = "#ec4899" if idx == 0 else ("#06b6d4" if idx == 1 else ("#a855f7" if idx == 2 else "#10b981"))
            border_color = f"rgba({int(accent_color[1:3],16)}, {int(accent_color[3:5],16)}, {int(accent_color[5:7],16)}, 0.4)"
            h = item.get("heuristic", {})
            overall = item.get("overall_score", 0.0)
            nov_val = max(0.0, round(h.get("novelty", 0.0), 1))

            card_html = f"""<div style="background: rgba(15, 23, 42, 0.85); border: 1px solid {border_color}; border-radius: 14px; padding: 10px;"><div style="font-size:0.7rem; font-weight:800; color:{accent_color}; text-transform:uppercase; font-family:'JetBrains Mono', monospace;">{item['label']}</div><div style="font-size:1.8rem; font-weight:800; color:#10b981; margin:4px 0 8px 0; line-height:1;">{overall}<span style="font-size:0.8rem; color:#94a3b8; font-weight:600;"> / 10</span></div><div class="compare-metric-row"><span>Clarity</span><strong style="color:#f8fafc; font-family:'JetBrains Mono', monospace;">{h.get('clarity', 0)}/10</strong></div><div class="compare-metric-row"><span>Specificity</span><strong style="color:#f8fafc; font-family:'JetBrains Mono', monospace;">{h.get('specificity', 0)}/10</strong></div><div class="compare-metric-row"><span>Novelty</span><strong style="color:#f8fafc; font-family:'JetBrains Mono', monospace;">{nov_val}/10</strong></div></div>"""

            with res_cols[idx]:
                st.markdown(card_html, unsafe_allow_html=True)

                rubric = item.get("llm_rubric")
                if rubric:
                    st.markdown("<p style='font-size:0.72rem; font-weight:700; color:#f8fafc; margin: var(--gap-2) 0 var(--gap-2) 0;'>Strategic Pillars:</p>", unsafe_allow_html=True)
                    for k, v in rubric.items():
                        score_val = v.get("score", 5.0) if isinstance(v, dict) else 5.0
                        just_val = v.get("justification", "") if isinstance(v, dict) else str(v)
                        formatted_key = k.replace("_", " ").title()
                        with st.expander(f"{formatted_key}: {score_val}/10"):
                            st.caption(just_val)

# ===========================================================================
# SCREEN 4: ADVISOR
# ===========================================================================
elif st.session_state["active_nav"] == "Advisor":
    st.markdown("<h2 style='font-size:1.1rem; font-weight:800; color:#f8fafc; margin:0;'>AI Consultant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#94a3b8; margin:0;'>Interactive consultation with active memory context.</p>", unsafe_allow_html=True)

    active_proposals = []

    if "result" in st.session_state and st.session_state["result"].get("extracted_text_full"):
        res = st.session_state["result"]
        t1_text = res["extracted_text_full"]
        score = res.get("overall_score", "N/A")
        rubric_json = json.dumps(res.get("llm_rubric", {}), indent=2)
        active_proposals.append(
            f"ACTIVE PROPOSAL (Evaluate Submission):\nOverall Score: {score}/10\n"
            f"Evaluated Pillar Metrics:\n{rubric_json}\n\nFull Proposal Text:\n{t1_text}"
        )

    if "comp_results" in st.session_state:
        for c in st.session_state["comp_results"]:
            lbl = c.get("label", "Proposal")
            txt = c.get("extracted_text_full", "")
            score = c.get("overall_score", "N/A")
            rubric_json = json.dumps(c.get("llm_rubric", {}), indent=2)
            if txt:
                active_proposals.append(
                    f"ACTIVE PROPOSAL [{lbl}] (Compare Submission):\nOverall Score: {score}/10\n"
                    f"Evaluated Pillar Metrics:\n{rubric_json}\n\nFull Proposal Text:\n{txt}"
                )

    if not active_proposals:
        st.markdown("""<div class="status-row-card"><div style="font-size:0.75rem; color:#f59e0b;">⚠️ No active proposal loaded in session memory!</div></div>""", unsafe_allow_html=True)
        st.caption("👉 Evaluate a proposal in the Evaluate tab or Compare tab to activate AI advisor.")
    else:
        st.markdown(f"""<div class="status-row-card"><div class="status-online-pill"><span class="status-dot-green"></span>Active Context Loaded: {len(active_proposals)} proposal(s) ready</div></div>""", unsafe_allow_html=True)

        if st.button("🧹 Reset Chat Memory", key="btn_reset_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.markdown(f'<div style="background:rgba(30, 41, 59, 0.8); padding:8px 10px; border-radius:10px; font-size:0.75rem; border:1px solid rgba(168, 85, 247, 0.3);">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:rgba(15, 23, 42, 0.85); padding:8px 10px; border-radius:10px; font-size:0.75rem; color:#c7c4d7; border:1px solid rgba(255,255,255,0.08);">{msg["content"]}</div>', unsafe_allow_html=True)

        user_query = st.chat_input("Ask a question about active proposals...")

        if user_query:
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            with st.spinner("Analyzing active context via Ollama..."):
                try:
                    joined_active_context = "\n\n====================\n\n".join(active_proposals)
                    prompt = f"""You are an expert enterprise innovation and technical proposal consultant.
Analyze the provided ACTIVE PROPOSAL TEXT and pre-evaluated pillar metrics below to answer the user's question accurately.

REASONING RULES TO PREVENT CONTRADICTIONS:
1. Explicitly differentiate between Execution Risk (technical complexity/implementation difficulty) and Commercial Viability (market size/competitive moat).
2. High technical complexity does NOT negate a high overall recommendation if the competitive moat and market size justify the investment. Always explain this balance clearly.
3. Keep market dynamics consistent across responses. Do not alter competitive intensity statements between queries; cite the evaluated pillar scores directly.

ACTIVE PROPOSAL TEXT & METRICS:
{joined_active_context}

User Question: {user_query}"""

                    res = ollama_client.chat.completions.create(
                        model="llama3",
                        messages=[
                            {"role": "system", "content": "You are a professional proposal advisor."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0
                    )

                    bot_reply = res.choices[0].message.content
                except Exception as e:
                    bot_reply = f"Error during query processing: {e}"

            st.session_state["chat_history"].append({"role": "assistant", "content": bot_reply})
            st.rerun()

# ===========================================================================
# STICKY BOTTOM NAVIGATION BAR
# ===========================================================================
with st.container(key="bottom_nav_wrap"):
    nav_cols = st.columns(4)
    nav_items = [
        ("Dashboard", "Dashboard", nav_cols[0]),
        ("Evaluate", "Evaluate", nav_cols[1]),
        ("Compare", "Compare", nav_cols[2]),
        ("Advisor", "Advisor", nav_cols[3]),
    ]
    for screen_name, label, col in nav_items:
        with col:
            is_active = st.session_state["active_nav"] == screen_name
            if st.button(
                label,
                key=f"nav_btn_{screen_name.lower()}",
                type="primary" if is_active else "secondary",
            ):
                navigate_to(screen_name)
                st.rerun()