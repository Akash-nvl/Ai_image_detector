"""
AI Image Forensics Studio
Intelligent Image Authenticity Analyzer
Streamlit application powered by a trained MobileNetV2 (CIFAKE) model.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from model_utils import (
    CLASS_INDICES,
    IMG_SIZE,
    INDEX_TO_CLASS,
    MODEL_FILENAME,
    RECORDED_TEST_ACCURACY,
    format_file_size,
    get_image_meta,
    load_image_rgb,
    load_model,
    model_status,
    predict,
    validate_image,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Image Forensics Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – dark premium theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-deep: #0b1220;
    --bg-card: #121a2b;
    --bg-card-hover: #182235;
    --border: #1e2a40;
    --cyan: #22d3ee;
    --cyan-dim: #0891b2;
    --violet: #a78bfa;
    --violet-dim: #7c3aed;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --radius: 14px;
    --shadow: 0 8px 32px rgba(0,0,0,0.35);
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0b1220 0%, #0f172a 50%, #111827 100%) !important;
    color: var(--text-primary);
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0b1220 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Cards */
.glass-card {
    background: linear-gradient(145deg, rgba(18,26,43,0.95), rgba(15,23,42,0.9));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.5rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.glass-card:hover {
    border-color: rgba(34,211,238,0.25);
}

.feature-card {
    background: linear-gradient(145deg, #121a2b, #0f172a);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    text-align: center;
    height: 100%;
    transition: border-color 0.2s ease;
}
.feature-card:hover {
    border-color: rgba(167,139,250,0.4);
}
.feature-icon {
    font-size: 1.75rem;
    margin-bottom: 0.5rem;
}
.feature-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
    margin-bottom: 0.35rem;
}
.feature-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* Typography */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f1f5f9 0%, #22d3ee 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 620px;
    line-height: 1.55;
    margin-bottom: 1.5rem;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.6rem;
}
.page-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}
.page-sub {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

/* Status pills */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
}
.status-ok {
    background: rgba(52,211,153,0.12);
    color: var(--green);
    border: 1px solid rgba(52,211,153,0.3);
}
.status-err {
    background: rgba(248,113,113,0.12);
    color: var(--red);
    border: 1px solid rgba(248,113,113,0.3);
}
.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.dot-ok { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot-err { background: var(--red); }

/* Result card */
.result-card {
    background: linear-gradient(145deg, #121a2b, #0f172a);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.75rem;
    text-align: center;
}
.result-class {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0.4rem 0;
}
.result-ai { color: var(--amber); }
.result-real { color: var(--green); }
.result-conf {
    font-size: 2.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--cyan);
    margin: 0.3rem 0;
}
.result-note {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-top: 1rem;
    line-height: 1.45;
    max-width: 420px;
    margin-left: auto;
    margin-right: auto;
}

/* Meta chips */
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0;
}
.meta-chip {
    background: rgba(30,42,64,0.8);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.65rem;
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}

/* Logo */
.logo-block {
    text-align: center;
    padding: 0.5rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.logo-mark {
    font-size: 1.6rem;
    margin-bottom: 0.2rem;
}
.logo-name {
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: -0.02em;
    color: var(--text-primary);
}
.logo-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}

/* Nav buttons */
div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-secondary) !important;
    font-weight: 500;
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    transition: all 0.15s ease;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(34,211,238,0.08) !important;
    border-color: rgba(34,211,238,0.2) !important;
    color: var(--cyan) !important;
}
div[data-testid="stSidebar"] .nav-active button {
    background: rgba(34,211,238,0.12) !important;
    border-color: rgba(34,211,238,0.35) !important;
    color: var(--cyan) !important;
}

/* Primary CTA */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0891b2, #7c3aed) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 16px rgba(8,145,178,0.3) !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
    filter: brightness(1.1);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card);
    border: 1px dashed rgba(34,211,238,0.35);
    border-radius: var(--radius);
    padding: 0.5rem;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--violet)) !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    color: var(--text-muted);
}
.empty-state .icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    opacity: 0.6;
}

/* Pipeline steps */
.pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    margin: 1rem 0;
}
.pipeline-step {
    background: rgba(30,42,64,0.9);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.35rem 0.7rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}
.pipeline-arrow {
    color: var(--text-muted);
    font-size: 0.8rem;
}

/* Divider */
.soft-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "page": "Dashboard",
        "history": [],
        "last_result": None,
        "last_image_meta": None,
        "uploaded_file_id": None,
        "prediction_done_for": None,
        "confirm_clear": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div class="logo-block">
                <div class="logo-mark">🔍</div>
                <div class="logo-name">AI Image Forensics Studio</div>
                <div class="logo-sub">Intelligent Image Authenticity Analyzer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        ok, msg = model_status()
        if ok:
            st.markdown(
                f'<div class="status-pill status-ok"><span class="dot dot-ok"></span> {msg}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="status-pill status-err"><span class="dot dot-err"></span> Model unavailable</div>',
                unsafe_allow_html=True,
            )
            st.caption(msg)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        nav_items = [
            ("Dashboard", "⌂"),
            ("Analyze Image", "▣"),
            ("Analysis History", "☰"),
            ("Model Information", "◎"),
            ("About Project", "ⓘ"),
        ]

        for label, icon in nav_items:
            active = st.session_state.page == label
            wrapper = "nav-active" if active else ""
            st.markdown(f'<div class="{wrapper}">', unsafe_allow_html=True)
            if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
                st.session_state.page = label
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
        n = len(st.session_state.history)
        st.caption(f"Session analyses: **{n}**")
        st.caption(f"Model: `{MODEL_FILENAME}`")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def page_dashboard():
    st.markdown('<div class="section-label">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">See Beyond the Pixels.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">'
        "Analyze an image with AI and explore whether it resembles the "
        "<strong>REAL</strong> or <strong>AI</strong> classes learned by our trained model."
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, _ = st.columns([1.2, 1.2, 2])
    with c1:
        if st.button("Analyze an Image", type="primary", use_container_width=True):
            st.session_state.page = "Analyze Image"
            st.rerun()
    with c2:
        if st.button("Explore Model", use_container_width=True):
            st.session_state.page = "Model Information"
            st.rerun()

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Feature cards
    f1, f2, f3 = st.columns(3)
    cards = [
        ("🤖", "AI-Powered Classification", "Binary classification using a fine-tuned MobileNetV2 network trained on CIFAKE."),
        ("⚡", "Instant Model Inference", "Local inference with cached model loading — no external API calls."),
        ("📊", "Transparent Prediction Results", "View class, confidence, raw sigmoid output, and full prediction details."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3], cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

    # Model overview
    st.markdown('<div class="section-label">Model Overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Architecture", "MobileNetV2")
    m2.metric("Dataset", "CIFAKE")
    m3.metric("Input Size", f"{IMG_SIZE}×{IMG_SIZE}")
    m4.metric("Test Accuracy", f"{RECORDED_TEST_ACCURACY * 100:.1f}%")

    st.caption(
        "Test accuracy is taken from the evaluation cell in `train_model.ipynb` "
        "(model.evaluate on the held-out test set)."
    )

    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

    # Recent analyses
    st.markdown('<div class="section-label">Recent Analyses</div>', unsafe_allow_html=True)
    history: List[Dict[str, Any]] = st.session_state.history
    if not history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">📭</div>
                <div>No analyses yet in this session.</div>
                <div style="font-size:0.85rem;margin-top:0.3rem">Upload an image to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        recent = list(reversed(history[-5:]))
        for rec in recent:
            cls = rec["predicted_class"]
            conf = rec["confidence_pct"]
            color = "var(--green)" if cls == "REAL" else "var(--amber)"
            st.markdown(
                f"""
                <div class="glass-card" style="padding:0.9rem 1.2rem;margin-bottom:0.5rem">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
                        <span style="font-weight:500">{rec['filename']}</span>
                        <span style="color:{color};font-weight:600">{cls}</span>
                        <span style="font-family:'JetBrains Mono',monospace;color:var(--cyan)">{conf}%</span>
                        <span style="color:var(--text-muted);font-size:0.8rem">{rec['timestamp']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Analyze Image
# ---------------------------------------------------------------------------

def page_analyze():
    st.markdown('<div class="section-label">Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Image Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Upload a JPG, JPEG, or PNG image and run inference with the trained MobileNetV2 model.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">Upload & Preview</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag and drop or browse",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="image_uploader",
            help="Accepted formats: JPG, JPEG, PNG",
        )

        if uploaded is not None:
            ok, msg = validate_image(uploaded)
            if not ok:
                st.error(msg)
            else:
                try:
                    img = load_image_rgb(uploaded)
                    meta = get_image_meta(uploaded, img)
                    st.session_state.last_image_meta = meta
                    st.session_state.uploaded_file_id = getattr(uploaded, "file_id", None) or uploaded.name

                    st.image(img, caption=meta["filename"], use_container_width=True)

                    st.markdown(
                        f"""
                        <div class="meta-row">
                            <span class="meta-chip">{meta['filename']}</span>
                            <span class="meta-chip">{meta['width']} × {meta['height']} px</span>
                            <span class="meta-chip">{format_file_size(meta['size_bytes'])}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if st.button("Clear image", key="clear_img"):
                        st.session_state.last_result = None
                        st.session_state.last_image_meta = None
                        st.session_state.prediction_done_for = None
                        st.session_state.uploaded_file_id = None
                        st.rerun()

                except Exception:
                    st.error("Failed to open the image. Please try another file.")
        else:
            st.markdown(
                """
                <div class="empty-state glass-card">
                    <div class="icon">🖼️</div>
                    <div>No image selected</div>
                    <div style="font-size:0.85rem;margin-top:0.3rem">
                        Drop a file above to preview it here.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-label">Prediction Controls</div>', unsafe_allow_html=True)

        if uploaded is None:
            st.info("Upload an image on the left to enable analysis.")
            return

        ok, msg = validate_image(uploaded)
        if not ok:
            st.warning(msg)
            return

        meta = st.session_state.get("last_image_meta") or {}
        st.markdown(f"**Selected:** `{meta.get('filename', uploaded.name)}`")

        run = st.button("Analyze Image", type="primary", use_container_width=True, key="run_predict")

        file_id = st.session_state.get("uploaded_file_id")
        already_done = (
            st.session_state.prediction_done_for == file_id
            and st.session_state.last_result is not None
        )

        if run or already_done:
            if run:
                with st.spinner("Running inference…"):
                    try:
                        model = load_model()
                        img = load_image_rgb(uploaded)
                        result = predict(model, img)
                        st.session_state.last_result = result
                        st.session_state.prediction_done_for = file_id

                        # Append to history
                        st.session_state.history.append(
                            {
                                "filename": meta.get("filename", uploaded.name),
                                "predicted_class": result["predicted_class"],
                                "confidence_pct": result["confidence_pct"],
                                "timestamp": result["timestamp"],
                                "raw_sigmoid": result["raw_sigmoid"],
                            }
                        )
                    except Exception as exc:
                        st.error(f"Inference failed: {exc}")
                        return

            result = st.session_state.last_result
            if result is None:
                return

            cls = result["predicted_class"]
            conf = result["confidence_pct"]
            cls_css = "result-real" if cls == "REAL" else "result-ai"
            label_desc = (
                "The model associates this image more strongly with the REAL class from the CIFAKE training set."
                if cls == "REAL"
                else "The model associates this image more strongly with the AI class from the CIFAKE training set."
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="section-label">Predicted Class</div>
                    <div class="result-class {cls_css}">{cls}</div>
                    <div class="section-label" style="margin-top:1rem">Confidence</div>
                    <div class="result-conf">{conf}%</div>
                    <div class="result-note">
                        {label_desc}<br><br>
                        This score represents the model's confidence in its predicted class.
                        It is not proof of an image's origin or authenticity.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(conf / 100.0, 1.0))

            with st.expander("Prediction Details", expanded=False):
                st.markdown(
                    f"""
                    | Field | Value |
                    |---|---|
                    | Model architecture | MobileNetV2 |
                    | Input image dimensions | {meta.get('width','?')} × {meta.get('height','?')} px |
                    | Model input dimensions | {IMG_SIZE} × {IMG_SIZE} px |
                    | Raw sigmoid output | `{result['raw_sigmoid']:.6f}` |
                    | Predicted class index | `{result['predicted_index']}` ({cls}) |
                    | P(REAL) | `{result['prob_real']:.4f}` |
                    | P(AI) | `{result['prob_ai']:.4f}` |
                    | Predicted-class confidence | `{conf}%` |
                    | Inference status | {result['status']} |
                    | Timestamp | {result['timestamp']} |
                    """
                )

            with st.expander("Understanding the Result", expanded=False):
                st.markdown(
                    f"""
                    The classifier was trained on the **CIFAKE** dataset with two classes:

                    - **AI** (index 0) — synthetic / AI-generated images in the training set  
                    - **REAL** (index 1) — photographic images in the training set  

                    The network outputs a single sigmoid value \\(p\\) interpreted as  
                    \\(P(\\text{{REAL}})\\).  

                    - If \\(p \\ge 0.5\\) → predicted class **REAL**, confidence = \\(p\\)  
                    - If \\(p < 0.5\\) → predicted class **AI**, confidence = \\(1 - p\\)  

                    For this image the model produced \\(p = {result['raw_sigmoid']:.4f}\\),  
                    so the predicted class is **{cls}** with confidence **{conf}%**.

                    > **Note:** This is a statistical association with the training distribution.  
                    > It does not identify the generator, detect local manipulations, or  
                    > verify file metadata.
                    """
                )

            if st.button("Analyze another image", key="another"):
                st.session_state.last_result = None
                st.session_state.prediction_done_for = None
                st.session_state.last_image_meta = None
                st.rerun()


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def page_history():
    st.markdown('<div class="section-label">Session</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analysis History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Predictions made during this browser session. Images are not stored permanently.</div>',
        unsafe_allow_html=True,
    )

    history: List[Dict[str, Any]] = st.session_state.history

    if not history:
        st.markdown(
            """
            <div class="empty-state glass-card">
                <div class="icon">📋</div>
                <div>History is empty</div>
                <div style="font-size:0.85rem;margin-top:0.3rem">
                    Completed analyses will appear here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    import pandas as pd

    df = pd.DataFrame(history)
    df = df.rename(
        columns={
            "filename": "Filename",
            "predicted_class": "Predicted Class",
            "confidence_pct": "Confidence (%)",
            "timestamp": "Timestamp",
            "raw_sigmoid": "Raw Sigmoid",
        }
    )
    # Show most recent first
    df = df.iloc[::-1].reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    if not st.session_state.confirm_clear:
        if st.button("Clear history", key="clear_hist"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("Are you sure you want to clear the analysis history?")
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            if st.button("Yes, clear", type="primary", key="confirm_yes"):
                st.session_state.history = []
                st.session_state.confirm_clear = False
                st.rerun()
        with c2:
            if st.button("Cancel", key="confirm_no"):
                st.session_state.confirm_clear = False
                st.rerun()


# ---------------------------------------------------------------------------
# Model Information
# ---------------------------------------------------------------------------

def page_model_info():
    st.markdown('<div class="section-label">Technical</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">Details of the trained MobileNetV2 classifier used by this application.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-label">Specification</div>
                <table style="width:100%;font-size:0.9rem;color:var(--text-secondary)">
                    <tr><td style="padding:0.3rem 0">Model name</td><td style="text-align:right;color:var(--text-primary)"><strong>MobileNetV2</strong></td></tr>
                    <tr><td style="padding:0.3rem 0">Dataset</td><td style="text-align:right;color:var(--text-primary)"><strong>CIFAKE</strong></td></tr>
                    <tr><td style="padding:0.3rem 0">Task</td><td style="text-align:right;color:var(--text-primary)">Binary classification</td></tr>
                    <tr><td style="padding:0.3rem 0">Input size</td><td style="text-align:right;color:var(--text-primary)">224 × 224 px</td></tr>
                    <tr><td style="padding:0.3rem 0">Output activation</td><td style="text-align:right;color:var(--text-primary)">Sigmoid</td></tr>
                    <tr><td style="padding:0.3rem 0">Output units</td><td style="text-align:right;color:var(--text-primary)">1</td></tr>
                    <tr><td style="padding:0.3rem 0">Saved model</td><td style="text-align:right;color:var(--text-primary)"><code>cifake_mobilenetv2.keras</code></td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-label">Architecture (from training)</div>
                <ul style="color:var(--text-secondary);font-size:0.9rem;line-height:1.7;padding-left:1.2rem;margin:0">
                    <li>MobileNetV2 base (ImageNet weights, frozen)</li>
                    <li>GlobalAveragePooling2D</li>
                    <li>BatchNormalization</li>
                    <li>Dense(128, ReLU)</li>
                    <li>Dropout(0.5)</li>
                    <li>Dense(1, Sigmoid)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label" style="margin-top:1rem">Class Mapping</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        From `train_data.class_indices` in the training notebook:

        | Class name | Index |
        |---|---|
        | **AI** | `{CLASS_INDICES['AI']}` |
        | **REAL** | `{CLASS_INDICES['REAL']}` |

        Sigmoid output \\(p\\) = probability of class index 1 (**REAL**).
        """
    )

    st.markdown('<div class="section-label" style="margin-top:1rem">Prediction Pipeline</div>', unsafe_allow_html=True)
    steps = [
        "Image Upload",
        "Validation",
        "Resize 224×224",
        "MobileNetV2 preprocess_input",
        "Model Inference",
        "Class Mapping",
        "Confidence Calc",
        "Results Display",
    ]
    parts = []
    for i, s in enumerate(steps):
        parts.append(f'<span class="pipeline-step">{s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="pipeline-arrow">→</span>')
    st.markdown(f'<div class="pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1rem">Recorded Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        Metrics below are taken from the evaluation output of `train_model.ipynb`.  
        Training ran for 15 epochs; early stopping restored weights from the best epoch.

        | Metric | Value |
        |---|---|
        | Test accuracy | **{RECORDED_TEST_ACCURACY * 100:.2f}%** |
        | Test loss | 0.2248 |
        | Final val accuracy (epoch 15) | 91.93% |
        | Final train accuracy (epoch 15) | ~89.98% |
        """
    )
    st.caption(
        "Training and validation curves were plotted in the notebook; "
        "they are not re-generated here. Refer to the notebook for the full history."
    )


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

def page_about():
    st.markdown('<div class="section-label">Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">About AI Image Forensics Studio</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">Overview</div>
            <p style="color:var(--text-secondary);line-height:1.6;margin:0">
                AI Image Forensics Studio is a local Streamlit application that applies a
                trained MobileNetV2 binary classifier to uploaded images. The model was
                trained on the CIFAKE dataset to distinguish images labeled <strong>AI</strong>
                from those labeled <strong>REAL</strong> in that dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">Objective</div>
            <p style="color:var(--text-secondary);line-height:1.6;margin:0">
                Provide a transparent, educational interface for inspecting how a
                convolutional neural network assigns an uploaded image to one of two
                training classes, along with the associated confidence score.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">Technologies</div>
            <ul style="color:var(--text-secondary);line-height:1.7;margin:0;padding-left:1.2rem">
                <li><strong>Python</strong> — application language</li>
                <li><strong>Streamlit</strong> — interactive web UI</li>
                <li><strong>TensorFlow / Keras</strong> — model loading and inference</li>
                <li><strong>Pillow</strong> — image decoding and resizing</li>
                <li><strong>NumPy</strong> — array operations</li>
                <li><strong>MobileNetV2</strong> — pretrained backbone (ImageNet)</li>
                <li><strong>CIFAKE</strong> — training / evaluation dataset</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">Workflow</div>
            <ol style="color:var(--text-secondary);line-height:1.7;margin:0;padding-left:1.2rem">
                <li>User uploads a JPG / PNG image.</li>
                <li>Image is validated, converted to RGB, and resized to 224×224.</li>
                <li>MobileNetV2 <code>preprocess_input</code> is applied (same as training).</li>
                <li>The cached Keras model performs a forward pass.</li>
                <li>Sigmoid output is mapped to class AI or REAL with confidence.</li>
                <li>Results and details are displayed; a session history entry is stored.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">Current Limitations</div>
            <ul style="color:var(--text-secondary);line-height:1.7;margin:0;padding-left:1.2rem">
                <li>Binary decision only (AI vs REAL as defined by CIFAKE labels).</li>
                <li>No generator identification, deepfake localization, or metadata forensics.</li>
                <li>No visual explainability (e.g. Grad-CAM) in the current release.</li>
                <li>Performance depends on how similar the input is to the CIFAKE distribution.</li>
                <li>History is session-local and not persisted to disk.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-label">Possible Future Enhancements</div>
            <ul style="color:var(--text-secondary);line-height:1.7;margin:0;padding-left:1.2rem">
                <li>Grad-CAM or similar explainability overlays (not implemented yet).</li>
                <li>Additional evaluation metrics (precision, recall, F1, ROC-AUC).</li>
                <li>Robustness testing against common image transformations.</li>
                <li>Optional EXIF / metadata inspection panel.</li>
                <li>Batch analysis mode for multiple images.</li>
            </ul>
            <p style="color:var(--text-muted);font-size:0.85rem;margin:0.75rem 0 0 0">
                The items above are planned ideas only — they are not present in this version.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def main():
    render_sidebar()

    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Analyze Image":
        page_analyze()
    elif page == "Analysis History":
        page_history()
    elif page == "Model Information":
        page_model_info()
    elif page == "About Project":
        page_about()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()