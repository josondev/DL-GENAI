"""
Messy Mashup — AST Genre Classifier
Streamlit deployment app for final-code.ipynb
Run: streamlit run app.py
"""

import os
import io
import random
import tempfile
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
from transformers import ASTForAudioClassification, ASTFeatureExtractor

# ─────────────────────────── CONFIG ───────────────────────────────────────────

SR = 16000
CHUNK_DURATION = 10
OVERLAP_DURATION = 2
CHUNK_LEN = SR * CHUNK_DURATION
STEP = SR * (CHUNK_DURATION - OVERLAP_DURATION)
FOLDS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
IDX2GENRE = {i: g for i, g in enumerate(GENRES)}

GENRE_EMOJI = {
    "blues": "🎸", "classical": "🎻", "country": "🤠", "disco": "🪩",
    "hiphop": "🎤", "jazz": "🎷", "metal": "🤘", "pop": "🌟",
    "reggae": "🌴", "rock": "⚡",
}

GENRE_COLOR = {
    "blues": "#1E90FF", "classical": "#DAA520", "country": "#8B4513",
    "disco": "#FF69B4", "hiphop": "#9400D3", "jazz": "#20B2AA",
    "metal": "#B22222", "pop": "#FF6347", "reggae": "#228B22", "rock": "#FF8C00",
}

MODEL_DIR = os.environ.get("MODEL_DIR", ".")   # set env var or drop .pth files next to app.py

# ─────────────────────────── PAGE SETUP ───────────────────────────────────────

st.set_page_config(
    page_title="Messy Mashup | Genre AI",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background: #0a0a0f;
    color: #e8e4d8;
  }

  .stApp { background: #0a0a0f; }

  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 7vw, 4.5rem);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.0;
    background: linear-gradient(135deg, #f5e6c8 0%, #e8b86d 50%, #c97b2a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
  }

  .hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #6b6456;
    margin-bottom: 2.5rem;
  }

  .card {
    background: #111118;
    border: 1px solid #1e1e2a;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
  }

  .genre-badge {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    border-radius: 50px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
  }

  .confidence-bar-outer {
    background: #1a1a24;
    border-radius: 4px;
    height: 7px;
    width: 100%;
    margin: 4px 0 10px 0;
    overflow: hidden;
  }

  .confidence-bar-inner {
    height: 7px;
    border-radius: 4px;
    transition: width 0.6s ease;
  }

  .label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #5a5469;
    margin-bottom: 2px;
  }

  .model-pill {
    display: inline-block;
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 0.68rem;
    color: #6b6456;
    margin-right: 4px;
  }

  .stFileUploader > div { border-color: #2a2a38 !important; }
  .stFileUploader label { color: #e8e4d8 !important; }

  div[data-testid="stAudio"] { margin-top: 0.5rem; }

  footer { visibility: hidden; }
  #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── HELPERS ──────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_feature_extractor():
    return ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )


@st.cache_resource(show_spinner=False)
def load_models(model_dir: str, n_folds: int = FOLDS):
    """Load fine-tuned fold checkpoints. Falls back to zero-shot if not found."""
    feature_extractor = load_feature_extractor()
    models = []
    loaded_paths = []

    for fold in range(n_folds):
        path = os.path.join(model_dir, f"model_fold{fold}.pth")
        m = ASTForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_labels=10,
            ignore_mismatched_sizes=True,
        )
        if os.path.exists(path):
            try:
                state = torch.load(path, map_location=DEVICE, weights_only=True)
                m.load_state_dict(state)
                loaded_paths.append(f"fold{fold} ✓")
            except Exception as e:
                loaded_paths.append(f"fold{fold} ✗")
        else:
            loaded_paths.append(f"fold{fold} (base)")

        m.eval().to(DEVICE)
        models.append(m)

    return models, feature_extractor, loaded_paths


def load_waveform(file_bytes: bytes) -> torch.Tensor:
    """Load audio bytes → mono waveform at SR."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        waveform, sr = librosa.load(tmp_path, sr=SR, mono=True)
        waveform = torch.from_numpy(waveform).float()
    finally:
        os.unlink(tmp_path)
    return waveform


@torch.no_grad()
def predict(waveform: torch.Tensor, models, feature_extractor) -> dict:
    """Chunked ensemble inference → softmax probability dict."""
    if len(waveform) < CHUNK_LEN:
        waveform = F.pad(waveform, (0, CHUNK_LEN - len(waveform)))

    length = len(waveform)
    num_chunks = min(int(np.ceil((length - CHUNK_LEN) / STEP)) + 1, 10)
    chunks = []
    for i in range(num_chunks):
        start = i * STEP
        end = start + CHUNK_LEN
        chunk = waveform[start:end]
        if len(chunk) < CHUNK_LEN:
            chunk = F.pad(chunk, (0, CHUNK_LEN - len(chunk)))
        chunks.append(chunk)

    logits_sum = torch.zeros(1, len(GENRES))
    total = 0

    for chunk in chunks:
        inputs = feature_extractor(
            chunk.numpy(), sampling_rate=SR, return_tensors="pt"
        )
        xb = inputs["input_values"].to(DEVICE)
        for model in models:
            out = model(input_values=xb)
            logits_sum += out.logits.cpu()
            total += 1

    avg = logits_sum / total
    probs = torch.softmax(avg, dim=-1).squeeze().tolist()
    return {IDX2GENRE[i]: round(p * 100, 2) for i, p in enumerate(probs)}


def render_results(probs: dict):
    top_genre = max(probs, key=probs.get)
    top_conf = probs[top_genre]
    color = GENRE_COLOR[top_genre]
    emoji = GENRE_EMOJI[top_genre]

    st.markdown(f"""
    <div class="card" style="border-color:{color}33; text-align:center;">
      <div style="font-size:3.5rem; margin-bottom:0.2rem">{emoji}</div>
      <div class="genre-badge" style="background:{color}22; color:{color}; border:2px solid {color}55;">
        {top_genre.upper()}
      </div>
      <div style="font-size:0.8rem; color:#6b6456; margin-top:0.4rem; font-family:'Space Mono',monospace;">
        {top_conf:.1f}% confidence
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### All Genres")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for genre, conf in sorted_probs:
        c = GENRE_COLOR[genre]
        st.markdown(f"""
        <div class="label-row"><span>{GENRE_EMOJI[genre]} {genre}</span><span>{conf:.1f}%</span></div>
        <div class="confidence-bar-outer">
          <div class="confidence-bar-inner" style="width:{conf}%;background:{c};"></div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────── UI ───────────────────────────────────────────────

st.markdown('<div class="hero-title">MESSY MASHUP</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AST · 4-Fold Ensemble · 10 Genres</div>', unsafe_allow_html=True)

# Load models
with st.spinner("Loading model weights…"):
    models, feature_extractor, loaded_paths = load_models(MODEL_DIR)

st.markdown(
    " ".join(f'<span class="model-pill">{p}</span>' for p in loaded_paths),
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# Upload
audio_file = st.file_uploader(
    "Drop an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Upload a music mashup or any audio clip (WAV recommended)",
)
st.markdown("</div>", unsafe_allow_html=True)

if audio_file is not None:
    file_bytes = audio_file.read()
    st.audio(file_bytes, format=audio_file.type)

    with st.spinner("Analysing audio…"):
        try:
            waveform = load_waveform(file_bytes)
            duration_s = len(waveform) / SR
            st.caption(
                f"**{audio_file.name}** · {duration_s:.1f}s · "
                f"{len(waveform):,} samples · {DEVICE.upper()}"
            )
            probs = predict(waveform, models, feature_extractor)
        except Exception as e:
            st.error(f"Could not process audio: {e}")
            probs = None

    if probs:
        st.markdown("---")
        render_results(probs)
