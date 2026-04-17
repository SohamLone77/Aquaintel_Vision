import base64
import html
import io
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from production_detector import UnderwaterThreatDetector as ProductionThreatDetector
from utils.config_loader import load_runtime_config
from utils.gpu import configure_tensorflow_device

APP_TITLE = "AquaIntel Vision"
APP_TAGLINE = "Production-grade enhancement & threat analysis for underwater imagery."
APP_SUBTITLE = "Enhance images, process video, and compare training runs with confidence."
APP_VERSION = "2026.04"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — deep-ocean dark design system
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;600;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design tokens ───────────────────────────────────────────────────── */
:root {
    --bg-deep:      #020c18;
    --bg-panel:     #081428;
    --bg-surface:   #0d1f38;
    --brand:        #00c8ff;
    --brand-dim:    rgba(0, 200, 255, 0.15);
    --brand-glow:   rgba(0, 200, 255, 0.35);
    --accent2:      #7c3aed;
    --accent2-dim:  rgba(124, 58, 237, 0.15);
    --danger:       #ff4d6a;
    --success:      #10b981;
    --gold:         #f5c518;
    --text:         #e2eaf5;
    --text-muted:   #6b8aad;
    --border:       rgba(0, 200, 255, 0.12);
    --border-hover: rgba(0, 200, 255, 0.42);
}

/* ── Base ────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

.stApp {
    background: radial-gradient(ellipse 120% 80% at 20% -10%, #051630 0%, #020c18 55%, #030f1e 100%);
    min-height: 100vh;
}

/* Subtle grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,200,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,.025) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-panel); }
::-webkit-scrollbar-thumb { background: var(--brand-dim); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--brand); }

/* ── Typography ─────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 0.04em;
    color: var(--text) !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #061020 0%, #030c1a 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px rgba(0, 200, 255, 0.06);
}

div[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-muted);
}

div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--brand) !important;
}

/* Sidebar expander headers */
div[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600;
    transition: border-color 0.2s;
}

div[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
    border-color: var(--brand-glow) !important;
}

div[data-testid="stSidebar"] .streamlit-expanderContent {
    background: rgba(13, 31, 56, 0.5) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Main content area ──────────────────────────────────────────────── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #00c8ff22 0%, #7c3aed22 100%) !important;
    color: var(--brand) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.25s ease !important;
    backdrop-filter: blur(8px) !important;
}

.stButton > button:hover {
    border-color: var(--brand) !important;
    box-shadow: 0 0 20px var(--brand-glow), inset 0 0 20px rgba(0,200,255,0.05) !important;
    transform: translateY(-1px) !important;
    color: #fff !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,200,255,0.25) 0%, rgba(124,58,237,0.25) 100%) !important;
    border-color: var(--brand) !important;
    color: #fff !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 28px var(--brand-glow) !important;
}

/* ── Download button ─────────────────────────────────────────────────── */
.stDownloadButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, rgba(16,185,129,0.20) 0%, rgba(0,200,255,0.15) 100%) !important;
    color: var(--success) !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
}

.stDownloadButton > button:hover {
    border-color: var(--success) !important;
    box-shadow: 0 0 20px rgba(16,185,129,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 4px !important;
    gap: 2px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
    font-size: 0.85rem !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--brand) !important;
    background: var(--brand-dim) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--brand) !important;
    background: var(--brand-dim) !important;
    box-shadow: 0 0 12px rgba(0,200,255,0.2) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s !important;
}

.stSelectbox > div > div:hover,
.stTextInput > div > div > input:focus {
    border-color: var(--brand) !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────── */
.stSlider > div > div > div > div {
    background: var(--brand) !important;
}

.stSlider > div > div > div {
    background: var(--bg-surface) !important;
}

/* ── Alerts ──────────────────────────────────────────────────────────── */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stAlert"][kind="info"],
.stInfo {
    background: rgba(0, 200, 255, 0.07) !important;
    border-left: 3px solid var(--brand) !important;
    color: var(--text) !important;
}

div[data-testid="stAlert"][kind="success"],
.stSuccess {
    background: rgba(16, 185, 129, 0.08) !important;
    border-left: 3px solid var(--success) !important;
}

div[data-testid="stAlert"][kind="warning"],
.stWarning {
    background: rgba(245, 197, 24, 0.08) !important;
    border-left: 3px solid var(--gold) !important;
}

div[data-testid="stAlert"][kind="error"],
.stError {
    background: rgba(255, 77, 106, 0.08) !important;
    border-left: 3px solid var(--danger) !important;
}

/* ── File uploader ───────────────────────────────────────────────────── */
section[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    transition: border-color 0.25s !important;
}

section[data-testid="stFileUploader"]:hover {
    border-color: var(--brand) !important;
}

/* ── Dataframe ────────────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress bar ─────────────────────────────────────────────────────── */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--brand) 0%, var(--accent2) 100%) !important;
    border-radius: 999px !important;
}

/* ── Metrics ─────────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
}

div[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--brand) !important;
    font-size: 2rem !important;
}

/* ── Custom components ───────────────────────────────────────────────── */

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(8,20,40,0.95) 0%, rgba(13,31,56,0.90) 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    box-shadow: 0 0 40px rgba(0,200,255,0.08), inset 0 1px 0 rgba(0,200,255,0.12);
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 80% at 90% 50%, rgba(0,200,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-left { z-index: 1; }

.hero-glyph {
    font-size: 2.6rem;
    line-height: 1;
    margin-bottom: 0.3rem;
    filter: drop-shadow(0 0 12px rgba(0,200,255,0.5));
}

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.85rem;
    font-weight: 900;
    letter-spacing: 0.06em;
    background: linear-gradient(90deg, #00c8ff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}

.hero-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}

.hero-right {
    text-align: right;
    z-index: 1;
    flex-shrink: 0;
}

.version-badge {
    display: inline-block;
    background: var(--brand-dim);
    color: var(--brand);
    border: 1px solid rgba(0,200,255,0.3);
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.hero-clock {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.08em;
}

/* Stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.25s, box-shadow 0.25s;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--brand) 0%, var(--accent2) 100%);
    border-radius: 16px 16px 0 0;
}

.stat-card:hover {
    border-color: var(--brand-glow);
    box-shadow: 0 0 24px rgba(0,200,255,0.12);
}

.stat-icon {
    font-size: 2rem;
    line-height: 1;
    filter: drop-shadow(0 0 8px rgba(0,200,255,0.4));
}

.stat-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--brand);
    line-height: 1;
}

.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}

.section-icon {
    font-size: 1.5rem;
    filter: drop-shadow(0 0 8px rgba(0,200,255,0.5));
}

.section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.04em;
    margin: 0;
}

.section-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 0;
}

/* Panel / glass card */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* Image frame */
.image-frame {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.25s;
}

.image-frame:hover {
    border-color: var(--brand-glow);
}

.image-frame-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--brand);
    margin-bottom: 0.5rem;
}

/* Empty state */
.empty-state {
    background: var(--bg-panel);
    border: 1.5px dashed rgba(0,200,255,0.2);
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
    filter: drop-shadow(0 0 12px rgba(0,200,255,0.3));
}

.empty-state-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
}

.empty-state-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--text-muted);
    opacity: 0.7;
}

/* Winner card */
.winner-card {
    background: linear-gradient(135deg, rgba(245,197,24,0.12) 0%, rgba(245,197,24,0.05) 100%);
    border: 1.5px solid rgba(245,197,24,0.45);
    border-radius: 16px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 0 30px rgba(245,197,24,0.08);
}

.winner-trophy {
    font-size: 2.4rem;
    filter: drop-shadow(0 0 10px rgba(245,197,24,0.6));
    flex-shrink: 0;
}

.winner-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.2rem;
}

.winner-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.2rem;
}

.winner-score {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--gold);
    opacity: 0.85;
}

/* Detection badge */
.detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, rgba(255,77,106,0.2) 0%, rgba(255,77,106,0.1) 100%);
    border: 1px solid rgba(255,77,106,0.4);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--danger);
    margin: 0.75rem 0;
}

.detection-badge.no-detections {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.35);
    color: var(--success);
}

/* Metadata table */
.meta-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
}

.meta-table tr {
    border-bottom: 1px solid var(--border);
}

.meta-table tr:last-child {
    border-bottom: none;
}

.meta-table td {
    padding: 0.45rem 0.5rem;
    color: var(--text-muted);
}

.meta-table td:first-child {
    font-weight: 600;
    color: var(--brand);
    white-space: nowrap;
    padding-right: 1.2rem;
    width: 40%;
}

.meta-table td:last-child {
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Sidebar logo block */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.6rem 0.4rem 0.4rem;
    margin-bottom: 0.5rem;
}

.sidebar-logo-icon {
    font-size: 1.6rem;
    filter: drop-shadow(0 0 8px rgba(0,200,255,0.5));
}

.sidebar-logo-name {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00c8ff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.06em;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 0.75rem 0;
}

/* Chip / tag */
.chip {
    display: inline-block;
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.chip-brand {
    background: var(--brand-dim);
    color: var(--brand);
    border: 1px solid rgba(0,200,255,0.3);
}

.chip-accent2 {
    background: var(--accent2-dim);
    color: #a78bfa;
    border: 1px solid rgba(124,58,237,0.35);
}

.chip-success {
    background: rgba(16,185,129,0.12);
    color: var(--success);
    border: 1px solid rgba(16,185,129,0.35);
}

.chip-danger {
    background: rgba(255,77,106,0.12);
    color: var(--danger);
    border: 1px solid rgba(255,77,106,0.35);
}

/* Preset strip */
.preset-strip {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
}

.preset-item {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--text-muted);
}

.preset-item strong {
    color: var(--brand);
    font-weight: 700;
}

/* Timing badge */
.timing-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(124,58,237,0.18);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 999px;
    padding: 0.28rem 0.85rem;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #a78bfa;
    letter-spacing: 0.05em;
}

/* Image metadata card */
.img-meta-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.img-meta-item {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    text-align: center;
}

.img-meta-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--brand);
}

.img-meta-key {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-top: 0.1rem;
}

/* ETA strip */
.eta-strip {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(0,200,255,0.05);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.75rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--text-muted);
}

.eta-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--brand);
}

/* Config panel replacing st.json */
.config-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem;
    margin-top: 0.5rem;
}

.config-row {
    background: var(--bg-surface);
    border-radius: 8px;
    padding: 0.4rem 0.65rem;
    display: flex;
    flex-direction: column;
}

.config-key {
    font-size: 0.63rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    font-weight: 600;
}

.config-val {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    color: var(--text);
    margin-top: 0.1rem;
    word-break: break-all;
}

/* Footer */
.app-footer {
    margin-top: 3rem;
    padding: 1.2rem 2rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.footer-brand {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00c8ff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.1em;
}

.footer-meta {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--text-muted);
    display: flex;
    gap: 1.5rem;
}

.footer-link {
    color: var(--brand);
    text-decoration: none;
    transition: opacity 0.2s;
}

.footer-link:hover { opacity: 0.75; }

/* ── Image comparison slider ─────────────────────────────────────────── */
.compare-wrapper {
    position: relative;
    width: 100%;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border);
    cursor: col-resize;
    user-select: none;
}

.compare-label {
    position: absolute;
    top: 10px;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    z-index: 5;
}

.compare-label-left {
    left: 10px;
    background: rgba(0,200,255,0.85);
    color: #020c18;
}

.compare-label-right {
    right: 10px;
    background: rgba(16,185,129,0.85);
    color: #020c18;
}

/* ── Quality metric cards ─────────────────────────────────────────────── */
.quality-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}

.quality-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
}

.quality-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}

.quality-card.good::before  { background: var(--success); }
.quality-card.warn::before  { background: var(--gold); }
.quality-card.bad::before   { background: var(--danger); }
.quality-card.info::before  { background: var(--brand); }

.quality-icon  { font-size: 1.6rem; line-height: 1; margin-bottom: 0.4rem; }
.quality-value { font-family: 'Orbitron', sans-serif; font-size: 1.6rem; font-weight: 700; color: var(--brand); }
.quality-name  { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-top: 0.2rem; }
.quality-note  { font-size: 0.65rem; color: var(--text-muted); opacity: 0.75; margin-top: 0.15rem; }

/* ── Batch cards ─────────────────────────────────────────────────────── */
.batch-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
}

.batch-card-icon { font-size: 1.5rem; flex-shrink: 0; }
.batch-card-name { font-weight: 600; color: var(--text); }
.batch-card-sub  { font-size: 0.7rem; color: var(--text-muted); }
.batch-card-right { margin-left: auto; }

/* ── Analytics section ──────────────────────────────────────────────── */
.analytics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.anlt-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    text-align: center;
}

.anlt-value { font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 700; color: var(--brand); }
.anlt-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-top: 0.2rem; }

/* ── Live training tab ───────────────────────────────────────────────── */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(16,185,129,0.18);
    border: 1px solid rgba(16,185,129,0.4);
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--success);
    letter-spacing: 0.06em;
    margin-bottom: 0.75rem;
    animation: pulse-badge 2s infinite;
}

@keyframes pulse-badge {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

/* Mobile guard */
@media (max-width: 640px) {
    .stat-grid { grid-template-columns: 1fr; }
    .hero-title { font-size: 1.3rem; }
    .hero-banner { flex-direction: column; align-items: flex-start; }
    .img-meta-grid { grid-template-columns: 1fr 1fr; }
    .quality-grid { grid-template-columns: 1fr 1fr; }
    .analytics-row { grid-template-columns: 1fr 1fr; }
    .app-footer { flex-direction: column; align-items: flex-start; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Helper: HTML components (all user-controlled values are html.escaped)
# ---------------------------------------------------------------------------

_e = html.escape  # shorthand


def safe_stem(filename: str) -> str:
    """Return a sanitized stem safe for filenames and HTML injection."""
    stem = Path(filename).name
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    return stem[:128]


def stat_card_html(icon: str, value: str | int, label: str) -> str:
    return f"""
<div class="stat-card">
  <div class="stat-icon">{icon}</div>
  <div class="stat-content">
    <div class="stat-value">{_e(str(value))}</div>
    <div class="stat-label">{_e(str(label))}</div>
  </div>
</div>
"""


def section_header_html(icon: str, title: str, subtitle: str = "") -> str:
    sub = f'<p class="section-subtitle">{_e(subtitle)}</p>' if subtitle else ""
    return f"""
<div class="section-header">
  <span class="section-icon">{icon}</span>
  <div>
    <p class="section-title">{_e(title)}</p>
    {sub}
  </div>
</div>
"""


def empty_state_html(icon: str, title: str, desc: str) -> str:
    return f"""
<div class="empty-state">
  <div class="empty-state-icon">{icon}</div>
  <div class="empty-state-title">{_e(title)}</div>
  <div class="empty-state-desc">{_e(desc)}</div>
</div>
"""


def image_frame_html(label: str) -> str:
    return f'<div class="image-frame-label">{_e(label)}</div>'


def winner_card_html(run_name: str, score: float, metric: str) -> str:
    return f"""
<div class="winner-card">
  <div class="winner-trophy">🏆</div>
  <div>
    <div class="winner-label">Recommended Run</div>
    <div class="winner-name">{_e(run_name)}</div>
    <div class="winner-score">Score: {_e(f'{score:.6f}')} on {_e(metric)}</div>
  </div>
</div>
"""


def detection_badge_html(count: int) -> str:
    css_class = "detection-badge" if count > 0 else "detection-badge no-detections"
    icon = "⚠️" if count > 0 else "✅"
    label = f"{count} threat{'s' if count != 1 else ''} detected" if count > 0 else "No threats detected"
    return f'<div class="{css_class}">{icon}&nbsp;{label}</div>'


def chip_html(text: str, style: str = "brand") -> str:
    safe_style = re.sub(r"[^a-z0-9-]", "", style)
    return f'<span class="chip chip-{safe_style}">{_e(text)}</span>'


def timing_badge_html(elapsed_ms: float) -> str:
    if elapsed_ms < 1000:
        label = f"{elapsed_ms:.0f} ms"
    else:
        label = f"{elapsed_ms/1000:.2f} s"
    return f'<div class="timing-badge">⏱️&nbsp;Inference: {_e(label)}</div>'


def image_meta_card_html(width: int, height: int, channels: int, size_kb: float, fmt: str) -> str:
    items = [
        (str(width), "Width px"),
        (str(height), "Height px"),
        (str(channels), "Channels"),
        (f"{size_kb:.1f}", "Size KB"),
        (fmt.upper(), "Format"),
        (f"{width*height//1000}K", "Megapixels"),
    ]
    rows = "".join(
        f'<div class="img-meta-item"><div class="img-meta-value">{_e(v)}</div><div class="img-meta-key">{_e(k)}</div></div>'
        for v, k in items
    )
    return f'<div class="img-meta-grid">{rows}</div>'


def config_panel_html(data: dict) -> str:
    """Replace st.json with a styled config grid."""
    rows = ""
    for k, v in data.items():
        if v is None:
            display = '<span style="color:#6b8aad">--</span>'
        elif isinstance(v, float):
            display = _e(f"{v:.6g}")
        else:
            display = _e(str(v))
        rows += f'<div class="config-row"><span class="config-key">{_e(k)}</span><span class="config-val">{display}</span></div>'
    return f'<div class="config-grid">{rows}</div>'


def render_footer():
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC+5:30")
    st.markdown(
        f"""
        <div class="app-footer">
          <div class="footer-brand">🌊 {APP_TITLE} &middot; v{APP_VERSION}</div>
          <div class="footer-meta">
            <span>🕐 Rendered {_e(now)}</span>
            <span>🛡️ All security patches active</span>
            <span>📄 AquaIntel Vision {APP_VERSION}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


DEEP_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#081428",
    plot_bgcolor="#0d1f38",
    font=dict(family="Inter, sans-serif", color="#e2eaf5", size=12),
    xaxis=dict(
        gridcolor="rgba(0,200,255,0.08)",
        linecolor="rgba(0,200,255,0.12)",
        tickfont=dict(color="#6b8aad"),
        title_font=dict(color="#00c8ff"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,200,255,0.08)",
        linecolor="rgba(0,200,255,0.12)",
        tickfont=dict(color="#6b8aad"),
        title_font=dict(color="#00c8ff"),
    ),
    legend=dict(
        bgcolor="rgba(8,20,40,0.8)",
        bordercolor="rgba(0,200,255,0.2)",
        borderwidth=1,
        font=dict(color="#e2eaf5"),
    ),
    margin=dict(l=55, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor="#0d1f38",
        bordercolor="rgba(0,200,255,0.3)",
        font=dict(color="#e2eaf5"),
    ),
)


def apply_global_styles():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Feature 1 — Image Comparison Slider
# ---------------------------------------------------------------------------

def _img_to_b64(arr: np.ndarray) -> str:
    """Encode numpy RGB array to base64 PNG string for inline HTML."""
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image_comparison_slider(
    original: np.ndarray,
    enhanced: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Enhanced",
    height: int = 420,
    component_key: str = "",
) -> None:
    """Render a pure-JS drag comparison slider between two images."""
    b64_left = _img_to_b64(original)
    b64_right = _img_to_b64(enhanced)
    uid = f"cmp_{component_key or uuid.uuid4().hex[:8]}"

    html_src = f"""
    <style>
      #{uid}-wrap {{
        position: relative; width: 100%; height: {height}px;
        border-radius: 14px; overflow: hidden;
        border: 1px solid rgba(0,200,255,0.2);
        cursor: col-resize; user-select: none;
        background: #020c18;
      }}
      #{uid}-right, #{uid}-left {{
        position: absolute; top: 0; left: 0;
        width: 100%; height: 100%;
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
      }}
      #{uid}-right {{ background-image: url('data:image/png;base64,{b64_right}'); }}
      #{uid}-left  {{ background-image: url('data:image/png;base64,{b64_left}'); }}
      #{uid}-clip  {{
        position: absolute; top: 0; left: 0;
        width: 50%; height: 100%; overflow: hidden;
      }}
      #{uid}-clip #{uid}-left {{ width: 100vw; max-width: 2400px; }}
      #{uid}-divider {{
        position: absolute; top: 0; left: 50%;
        transform: translateX(-50%);
        width: 3px; height: 100%;
        background: rgba(0,200,255,0.8);
        box-shadow: 0 0 12px rgba(0,200,255,0.6);
        z-index: 4;
      }}
      #{uid}-handle {{
        position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 38px; height: 38px;
        background: rgba(0,200,255,0.9);
        border-radius: 50%;
        z-index: 5;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: bold; color: #020c18;
        box-shadow: 0 0 18px rgba(0,200,255,0.5);
        cursor: col-resize;
      }}
      .{uid}-badge {{
        position: absolute; top: 12px;
        font-family: 'Orbitron', monospace; font-size: 0.6rem;
        font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
        padding: 3px 8px; border-radius: 6px; z-index: 6;
      }}
      .{uid}-badge-l {{ left: 10px; background: rgba(0,200,255,0.85); color: #020c18; }}
      .{uid}-badge-r {{ right: 10px; background: rgba(16,185,129,0.85); color: #020c18; }}
    </style>
    <div id="{uid}-wrap">
      <div id="{uid}-right"></div>
      <div id="{uid}-clip"><div id="{uid}-left"></div></div>
      <div id="{uid}-divider"><div id="{uid}-handle">↔</div></div>
      <span class="{uid}-badge {uid}-badge-l">{label_left}</span>
      <span class="{uid}-badge {uid}-badge-r">{label_right}</span>
    </div>
    <script>
    (function(){{
      const wrap = document.getElementById('{uid}-wrap');
      const clip = document.getElementById('{uid}-clip');
      const div  = document.getElementById('{uid}-divider');
      let dragging = false;
      function setPos(x) {{
        const rect = wrap.getBoundingClientRect();
        let pct = Math.max(2, Math.min(98, (x - rect.left) / rect.width * 100));
        clip.style.width = pct + '%';
        div.style.left   = pct + '%';
      }}
      wrap.addEventListener('mousedown', e => {{ dragging = true; setPos(e.clientX); }});
      window.addEventListener('mousemove', e => {{ if (dragging) setPos(e.clientX); }});
      window.addEventListener('mouseup',   () => {{ dragging = false; }});
      wrap.addEventListener('touchstart',  e => {{ dragging = true; setPos(e.touches[0].clientX); }}, {{passive:true}});
      window.addEventListener('touchmove', e => {{ if (dragging) setPos(e.touches[0].clientX); }}, {{passive:true}});
      window.addEventListener('touchend',  () => {{ dragging = false; }});
    }})()
    </script>
    """
    import streamlit.components.v1 as components
    components.html(html_src, height=height + 10, scrolling=False)


# ---------------------------------------------------------------------------
# Feature 3 — PSNR / SSIM quality metrics
# ---------------------------------------------------------------------------

def compute_quality_metrics(original: np.ndarray, enhanced: np.ndarray) -> dict:
    """Compute PSNR, SSIM, mean brightness delta, and RMS contrast delta."""
    orig_f = original.astype(np.float64)
    enhn_f = enhanced.astype(np.float64)

    # PSNR
    if _SKIMAGE_AVAILABLE:
        psnr = float(_psnr_fn(original, enhanced, data_range=255))
        ssim_val = float(_ssim_fn(original, enhanced, channel_axis=-1, data_range=255))
    else:
        mse = np.mean((orig_f - enhn_f) ** 2)
        psnr = float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 100.0
        ssim_val = None

    brightness_orig = float(np.mean(original))
    brightness_enhn = float(np.mean(enhanced))
    delta_brightness = brightness_enhn - brightness_orig

    contrast_orig = float(np.std(original))
    contrast_enhn = float(np.std(enhanced))
    delta_contrast = contrast_enhn - contrast_orig

    return {
        "psnr": psnr,
        "ssim": ssim_val,
        "brightness_delta": delta_brightness,
        "contrast_delta": delta_contrast,
    }


def quality_cards_html(metrics: dict) -> str:
    """Render PSNR / SSIM / brightness / contrast as branded metric cards."""
    psnr = metrics["psnr"]
    ssim = metrics.get("ssim")
    bd   = metrics["brightness_delta"]
    cd   = metrics["contrast_delta"]

    # PSNR rating
    if psnr >= 35:   psnr_cls, psnr_note = "good", "Excellent"
    elif psnr >= 28: psnr_cls, psnr_note = "warn", "Good"
    else:            psnr_cls, psnr_note = "bad",  "Needs work"

    # SSIM rating
    if ssim is None:
        ssim_cls, ssim_disp, ssim_note = "info", "N/A", "skimage missing"
    elif ssim >= 0.90: ssim_cls, ssim_disp, ssim_note = "good", f"{ssim:.3f}", "High fidelity"
    elif ssim >= 0.75: ssim_cls, ssim_disp, ssim_note = "warn", f"{ssim:.3f}", "Moderate"
    else:              ssim_cls, ssim_disp, ssim_note = "bad",  f"{ssim:.3f}", "Low similarity"

    bd_sign  = "+" if bd >= 0 else ""
    cd_sign  = "+" if cd >= 0 else ""
    bd_cls   = "good" if abs(bd) < 20 else "warn"
    cd_cls   = "good" if cd >= 0 else "warn"

    cards = [
        ("good" if psnr_cls == "good" else psnr_cls, "📁", f"{psnr:.1f} dB", "PSNR", psnr_note),
        (ssim_cls,  "🎯", ssim_disp,             "SSIM",       ssim_note),
        (bd_cls,    "☀️", f"{bd_sign}{bd:.1f}", "Brightness Δ", "vs original"),
        (cd_cls,    "🔆", f"{cd_sign}{cd:.1f}", "Contrast Δ",  "vs original"),
    ]
    inner = "".join(
        f'<div class="quality-card {cls}">'
        f'<div class="quality-icon">{icon}</div>'
        f'<div class="quality-value">{_e(val)}</div>'
        f'<div class="quality-name">{_e(name)}</div>'
        f'<div class="quality-note">{_e(note)}</div>'
        f'</div>'
        for cls, icon, val, name, note in cards
    )
    return f'<div class="quality-grid">{inner}</div>'


# ---------------------------------------------------------------------------
# Feature 5 — Session Analytics
# ---------------------------------------------------------------------------

def _init_analytics():
    """Initialise session analytics store once."""
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = {
            "session_id": uuid.uuid4().hex[:8].upper(),
            "session_start": time.time(),
            "inferences": [],         # [{"ts", "model", "img_size", "elapsed_ms", "psnr"}]
            "video_jobs": [],          # [{"ts", "model", "frames", "elapsed_s", "fps"}]
            "detections_run": 0,
            "tabs_visited": set(),
        }


def _log_inference(model_name: str, img_size: int, elapsed_ms: float, psnr: float | None = None):
    _init_analytics()
    st.session_state["analytics"]["inferences"].append({
        "ts": time.time(), "model": model_name,
        "img_size": img_size, "elapsed_ms": elapsed_ms, "psnr": psnr,
    })


def _log_video_job(model_name: str, frames: int, elapsed_s: float):
    _init_analytics()
    fps = frames / elapsed_s if elapsed_s > 0 else 0.0
    st.session_state["analytics"]["video_jobs"].append({
        "ts": time.time(), "model": model_name,
        "frames": frames, "elapsed_s": elapsed_s, "fps": fps,
    })


def render_analytics_dashboard():
    """Render full analytics summary in a sidebar expander."""
    _init_analytics()
    a = st.session_state["analytics"]
    infs = a["inferences"]
    vids = a["video_jobs"]
    uptime_s = int(time.time() - a["session_start"])
    h, rem = divmod(uptime_s, 3600)
    m, s   = divmod(rem, 60)
    uptime_str = f"{h:02d}:{m:02d}:{s:02d}"

    avg_inf = sum(i["elapsed_ms"] for i in infs) / len(infs) if infs else 0
    total_frames = sum(v["frames"] for v in vids)
    avg_fps = sum(v["fps"] for v in vids) / len(vids) if vids else 0

    counts = [
        (len(infs), "Images Enhanced"),
        (f"{avg_inf:.0f} ms", "Avg Inference"),
        (total_frames, "Video Frames"),
        (f"{avg_fps:.1f}", "Avg FPS"),
    ]

    cards_html = "".join(
        f'<div class="anlt-card"><div class="anlt-value">{_e(str(v))}</div><div class="anlt-label">{_e(l)}</div></div>'
        for v, l in counts
    )
    st.markdown(f'<div class="analytics-row">{cards_html}</div>', unsafe_allow_html=True)
    st.caption(f"Session ID: `{a['session_id']}` · Uptime: `{uptime_str}`")

    if infs and _PLOTLY_AVAILABLE:
        df_inf = pd.DataFrame(infs)
        df_inf["ts_label"] = [f"#{i+1}" for i in range(len(df_inf))]
        fig = go.Figure(go.Bar(
            x=df_inf["ts_label"], y=df_inf["elapsed_ms"],
            marker_color="#00c8ff", opacity=0.85,
        ))
        fig.update_layout(**DEEP_PLOTLY_LAYOUT, height=200,
                          xaxis_title="Inference #", yaxis_title="ms",
                          title="Inference Latency (ms)",
                          title_font=dict(family="Orbitron,sans-serif", size=11, color="#00c8ff"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if vids and _PLOTLY_AVAILABLE:
        df_vid = pd.DataFrame(vids)
        df_vid["label"] = [f"Job #{i+1}" for i in range(len(df_vid))]
        fig2 = go.Figure(go.Bar(
            x=df_vid["label"], y=df_vid["fps"],
            marker_color="#a78bfa", opacity=0.85,
        ))
        fig2.update_layout(**DEEP_PLOTLY_LAYOUT, height=200,
                           xaxis_title="Video Job", yaxis_title="FPS",
                           title="Video Processing FPS per Job",
                           title_font=dict(family="Orbitron,sans-serif", size=11, color="#00c8ff"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Feature 2 — Batch Image Inference
# ---------------------------------------------------------------------------

def run_batch_inference_view(model_choice: Path, img_size: int):
    st.markdown(
        section_header_html("📦", "Batch Image Enhancement", "Upload up to 50 images — all enhanced in one shot and bundled as a ZIP."),
        unsafe_allow_html=True,
    )

    files = st.file_uploader(
        "Upload images (JPG / PNG / BMP — max 20 MB each)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if not files:
        st.markdown(
            empty_state_html("📦", "No Files Uploaded", "Select multiple images above to run batch enhancement."),
            unsafe_allow_html=True,
        )
        return

    MAX_BATCH = 50
    if len(files) > MAX_BATCH:
        st.warning(f"Batch limited to {MAX_BATCH} images. First {MAX_BATCH} will be processed.")
        files = files[:MAX_BATCH]

    st.markdown(
        f'<div class="image-frame-label">🗒️&nbsp; {len(files)} file{"s" if len(files)!=1 else ""} ready</div>',
        unsafe_allow_html=True,
    )

    apply_detail   = st.checkbox("Apply detail fusion",  value=True,  key="batch_detail")
    apply_sharpen  = st.checkbox("Apply sharpening",     value=True,  key="batch_sharpen")
    show_previews  = st.checkbox("Show side-by-side previews", value=True, key="batch_preview")

    if not st.button("⚡ Enhance All", type="primary", key="batch_enhance_btn"):
        return

    model = load_model(str(model_choice))
    progress = st.progress(0)
    status   = st.empty()

    # Derive the true expected input size from the model itself.
    # get_model_input_size() uses a filename regex that falls back to 128,
    # which causes a shape-mismatch when the actual model requires 256 (or other).
    try:
        actual_img_size = int(model.input_shape[1])
    except Exception:
        actual_img_size = img_size  # fall back to heuristic if shape query fails

    if actual_img_size != img_size:
        st.info(
            f"ℹ️ Model input size detected from weights: **{actual_img_size}×{actual_img_size}** "
            f"(filename heuristic said {img_size}). Using {actual_img_size}."
        )

    zip_buf = io.BytesIO()
    results  = []  # (name, original, enhanced)
    t_batch_start = time.perf_counter()

    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, f in enumerate(files):
            if f.size > 20 * 1024 * 1024:
                status.warning(f"⚠️ Skipping {f.name}: exceeds 20 MB.")
                continue

            f.seek(0)
            try:
                pil_img = Image.open(f).convert("RGB")
            except Exception:
                status.warning(f"⚠️ Skipping {f.name}: cannot decode.")
                continue

            orig_arr, inp, meta = preprocess_image(pil_img, actual_img_size)
            t0  = time.perf_counter()
            pred = run_inference(model, inp)
            t_ms = (time.perf_counter() - t0) * 1000

            h, w = orig_arr.shape[:2]
            enh = postprocess_prediction(pred, meta, (w, h))
            if apply_detail:
                enh = fuse_details_from_original(orig_arr, enh, 0.45, 0.4 if apply_sharpen else 0.0)

            stem = safe_stem(f.name)
            out_name = f"enhanced_{stem}.png"
            zf.writestr(out_name, image_to_download_bytes(enh))

            results.append((f.name, orig_arr, enh, t_ms))
            progress.progress(int((idx + 1) / len(files) * 100))
            status.markdown(
                f'<span style="font-family:\'Orbitron\',sans-serif;font-size:0.8rem;color:#00c8ff">'  
                f'Processed {idx+1}/{len(files)} — {f.name}</span>',
                unsafe_allow_html=True,
            )

    elapsed_batch = time.perf_counter() - t_batch_start
    avg_ms = sum(r[3] for r in results) / len(results) if results else 0

    # Log to analytics
    for r in results:
        _log_inference(model_choice.name, img_size, r[3])

    st.success(f"✅ Batch complete — {len(results)} images in {elapsed_batch:.1f}s (avg {avg_ms:.0f} ms/image).")

    zip_buf.seek(0)
    st.download_button(
        label="⬇️ Download All as ZIP",
        data=zip_buf,
        file_name="aquaintel_enhanced_batch.zip",
        mime="application/zip",
        key="batch_zip_download",
    )

    if show_previews and results:
        st.markdown(section_header_html("🖼️", "Batch Previews"), unsafe_allow_html=True)
        for name, orig, enh, ms in results[:12]:  # cap previews at 12
            with st.expander(f"🖼 {name}  ({ms:.0f} ms)", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(image_frame_html("🔵 Original"), unsafe_allow_html=True)
                    st.image(orig, use_container_width=True)
                with c2:
                    st.markdown(image_frame_html("✨ Enhanced"), unsafe_allow_html=True)
                    st.image(enh, use_container_width=True)


# ---------------------------------------------------------------------------
# Feature 4 — Live Training Dashboard
# ---------------------------------------------------------------------------

def run_live_training_view(logs_dir: Path):
    st.markdown(
        section_header_html("📡", "Live Training Monitor", "Auto-refreshing training metrics from CSV logs. Works while training runs in parallel."),
        unsafe_allow_html=True,
    )

    auto_refresh = st.checkbox("Auto-refresh every 10 s", value=False, key="live_auto_refresh")
    if st.button("🔄  Refresh Now", key="live_refresh_btn"):
        st.rerun()

    csv_files = sorted(logs_dir.glob("*_training.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        st.markdown(
            empty_state_html("📡", "No Training Logs Found", f"Place training CSV logs in {logs_dir}"),
            unsafe_allow_html=True,
        )
        if auto_refresh:
            time.sleep(10)
            st.rerun()
        return

    st.markdown(
        f'<span class="live-badge">● LIVE &nbsp;—&nbsp; {len(csv_files)} run{"s" if len(csv_files)!=1 else ""} found</span>',
        unsafe_allow_html=True,
    )

    run_names = [p.stem.replace("_training", "") for p in csv_files]
    selected_runs = st.multiselect(
        "Select runs to display", run_names,
        default=run_names[:min(3, len(run_names))],
        key="live_selected_runs",
    )

    metric_cols = st.columns(3)
    summary_rows = []

    figs_loss = go.Figure() if _PLOTLY_AVAILABLE else None
    figs_mae  = go.Figure() if _PLOTLY_AVAILABLE else None
    PALETTE   = ["#00c8ff","#a78bfa","#10b981","#f5c518","#ff4d6a","#fb923c"]

    for i, run_name in enumerate(selected_runs):
        csv_path = logs_dir / f"{run_name}_training.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue

        color = PALETTE[i % len(PALETTE)]
        last = df.iloc[-1]
        epoch_col = df.get("epoch", pd.Series(range(len(df))))

        summary_rows.append({
            "run": run_name,
            "epochs": int(len(df)),
            "last_loss": last.get("loss"),
            "last_val_loss": last.get("val_loss"),
            "best_val_loss": df["val_loss"].min() if "val_loss" in df else None,
            "last_val_mae": last.get("val_mae"),
        })

        if _PLOTLY_AVAILABLE and figs_loss is not None:
            if "loss" in df.columns:
                figs_loss.add_trace(go.Scatter(
                    x=epoch_col, y=df["loss"],
                    name=f"{run_name} train",
                    line=dict(color=color, width=2, dash="dot"),
                    mode="lines",
                ))
            if "val_loss" in df.columns:
                figs_loss.add_trace(go.Scatter(
                    x=epoch_col, y=df["val_loss"],
                    name=f"{run_name} val",
                    line=dict(color=color, width=2.5),
                    mode="lines+markers", marker=dict(size=3),
                ))
            if "val_mae" in df.columns and figs_mae is not None:
                figs_mae.add_trace(go.Scatter(
                    x=epoch_col, y=df["val_mae"],
                    name=f"{run_name} val_mae",
                    line=dict(color=color, width=2.5),
                    mode="lines+markers", marker=dict(size=3),
                ))

    if _PLOTLY_AVAILABLE and figs_loss is not None and figs_loss.data:
        figs_loss.update_layout(**DEEP_PLOTLY_LAYOUT, height=340,
                                xaxis_title="Epoch", yaxis_title="Loss",
                                title="Training & Validation Loss",
                                title_font=dict(family="Orbitron,sans-serif", size=13, color="#00c8ff"))
        st.plotly_chart(figs_loss, use_container_width=True)

    if _PLOTLY_AVAILABLE and figs_mae is not None and figs_mae.data:
        figs_mae.update_layout(**DEEP_PLOTLY_LAYOUT, height=280,
                               xaxis_title="Epoch", yaxis_title="Val MAE",
                               title="Validation MAE",
                               title_font=dict(family="Orbitron,sans-serif", size=13, color="#00c8ff"))
        st.plotly_chart(figs_mae, use_container_width=True)

    if summary_rows:
        st.markdown(section_header_html("📋", "Run Summary Table"), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            column_config={
                "best_val_loss": st.column_config.NumberColumn("Best Val Loss", format="%.5f"),
                "last_val_loss": st.column_config.NumberColumn("Last Val Loss", format="%.5f"),
                "last_val_mae":  st.column_config.NumberColumn("Last Val MAE",  format="%.5f"),
            },
        )

    if auto_refresh:
        time.sleep(10)
        st.rerun()


def render_hero():
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    st.markdown(
        f"""
        <div class="hero-banner">
          <div class="hero-left">
            <div class="hero-glyph">🌊</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-tagline">{APP_TAGLINE}</div>
          </div>
          <div class="hero-right">
            <div class="version-badge">v&nbsp;{APP_VERSION}</div>
            <div class="hero-clock">🕐 {_e(now)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def list_model_files(checkpoint_dir: Path):
    model_files = list(checkpoint_dir.glob("*.keras")) + list(checkpoint_dir.glob("*.h5"))
    return sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_registry(registry_path: Path):
    if not registry_path.exists():
        return {}
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_resource
def load_model(model_path_str: str):
    # Fix N3: Validate path before loading to prevent path-traversal via internal calls
    try:
        from utils.config_loader import load_runtime_config
        checkpoint_dir = Path(load_runtime_config().get("checkpoint_dir", "models/checkpoints")).resolve()
        resolved = Path(model_path_str).resolve()
        if not resolved.is_relative_to(checkpoint_dir):
            raise ValueError(f"Security: model path '{Path(model_path_str).name}' avoids checkpoint directory.")
    except Exception as exc:
        raise ValueError(f"Invalid model path: {exc}")
    return tf.keras.models.load_model(model_path_str, compile=False)


@st.cache_data
def get_model_input_size(model_path_str: str):
    # Fix N4: Prevent loading full model (costing GPU RAM and pickle risk) just to get the size
    import re
    stem = Path(model_path_str).stem
    match = re.search(r'_(128|256|384|512|1024)(?:_|$)', stem)
    if match:
        return int(match.group(1))
    return 128


def preprocess_image(uploaded: Image.Image, img_size: int):
    arr = np.array(uploaded.convert("RGB"))
    return preprocess_rgb_array(arr, img_size)


def preprocess_rgb_array(arr: np.ndarray, img_size: int):
    h, w = arr.shape[:2]

    # Letterbox to avoid aspect-ratio distortion before inference.
    scale = min(img_size / max(1, w), img_size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    x0 = (img_size - new_w) // 2
    y0 = (img_size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    x = canvas.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    meta = {"x0": x0, "y0": y0, "new_w": new_w, "new_h": new_h}
    return arr, x, meta


def postprocess_prediction(pred, meta: dict, output_size: tuple[int, int]):
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255.0).astype(np.uint8)

    x0 = int(meta["x0"])
    y0 = int(meta["y0"])
    new_w = int(meta["new_w"])
    new_h = int(meta["new_h"])

    cropped = pred[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)


def image_to_download_bytes(img_array: np.ndarray):
    image = Image.fromarray(img_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def fuse_details_from_original(
    original_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    detail_strength: float = 0.45,
    sharpen_amount: float = 0.4,
):
    """Recover high-frequency detail from the original while keeping enhanced color tone."""
    detail_strength = float(np.clip(detail_strength, 0.0, 1.0))
    sharpen_amount = float(np.clip(sharpen_amount, 0.0, 2.0))

    base = enhanced_rgb.astype(np.float32)
    orig = original_rgb.astype(np.float32)

    # High-frequency residual from original image.
    blur = cv2.GaussianBlur(orig, (0, 0), sigmaX=1.2, sigmaY=1.2)
    high_freq = orig - blur

    fused = base + (detail_strength * high_freq)

    if sharpen_amount > 0.0:
        # Mild unsharp mask on fused result.
        fused_blur = cv2.GaussianBlur(fused, (0, 0), sigmaX=1.0, sigmaY=1.0)
        fused = cv2.addWeighted(fused, 1.0 + sharpen_amount, fused_blur, -sharpen_amount, 0)

    return np.clip(fused, 0, 255).astype(np.uint8)


def run_inference(model, input_batch):
    pred = model.predict(input_batch, verbose=0)
    return pred[0]


def make_web_preview_video(source_path: Path):
    """Transcode to a browser-friendly H.264 MP4 when ffmpeg is available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = None

    if ffmpeg_path is None:
        return source_path, "FFmpeg not found (system or bundled). Preview may fail in some browsers for MPEG-4 output codecs."

    preview_path = source_path.with_name(f"{source_path.stem}_web.mp4")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(preview_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    except Exception:
        return source_path, "Could not transcode preview to H.264. Downloaded video is still available."

    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path, None

    return source_path, "Preview transcode output was empty. Downloaded video is still available."


def list_yolo_model_files():
    candidates = sorted(
        Path("runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Add pre-trained fallback as a virtual option.
    return ["yolov8n.pt"] + [str(p).replace("\\", "/") for p in candidates]


class StreamlitThreatDetectorAdapter:
    """Keep Streamlit view code stable while using the production detector API."""

    def __init__(self, detector: ProductionThreatDetector):
        self.detector = detector

    def detect_threats(self, image_bgr: np.ndarray):
        detections, annotated_bgr, _ = self.detector.detect(image_bgr)

        normalized = []
        for det in detections:
            normalized.append(
                {
                    "class": det.get("class_name", "unknown"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "threat_level": det.get("threat_level", "UNKNOWN"),
                    "bbox": det.get("bbox", []),
                    "timestamp": det.get("timestamp", ""),
                }
            )

        return normalized, annotated_bgr


@st.cache_resource
def load_threat_detector(
    enhancement_model_path: str,
    yolo_model_path: str,
    confidence_threshold: float,
    use_enhancement: bool,
):
    detector = ProductionThreatDetector(
        enhancement_model_path=enhancement_model_path,
        yolo_model_path=yolo_model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.45,
        enable_enhancement=use_enhancement,
    )
    return StreamlitThreatDetectorAdapter(detector)


def get_run_name_from_model_path(model_path: Path):
    stem = model_path.stem
    if stem.endswith("_best"):
        return stem[:-5]
    if stem.endswith("_final"):
        return stem[:-6]
    return stem


def infer_pipeline_type(run_name: str, run_meta: dict | None = None):
    """Infer whether a run belongs to standard or sharp pipeline."""
    lowered = run_name.lower()
    if lowered.startswith("sharp_") or "sharp" in lowered:
        return "sharp"

    if run_meta:
        cfg = run_meta.get("config", {})
        loss_type = str(cfg.get("loss_type", "")).lower()
        if loss_type == "sharp":
            return "sharp"

    return "standard"


def resolve_run_metadata(registry_data: dict, run_name: str, model_choice: Path):
    """Resolve run metadata by run key or artifact path match."""
    run_meta = registry_data.get(run_name)
    if run_meta:
        return run_meta

    model_name = model_choice.name.replace("\\", "/")
    for _, candidate in registry_data.items():
        artifacts = candidate.get("artifacts", {})
        for artifact_path in artifacts.values():
            if not artifact_path:
                continue
            artifact_name = str(artifact_path).replace("\\", "/").split("/")[-1]
            if artifact_name == model_name:
                return candidate

    return None


def build_history_fallback(run_name: str, logs_dir: Path, model_choice: Path):
    """Create minimal metadata from CSV history + file stats when registry entry is missing."""
    history_df = load_history(run_name, logs_dir)
    if history_df is None or history_df.empty:
        return None

    metrics = {
        "final_loss": float(history_df["loss"].iloc[-1]) if "loss" in history_df else None,
        "final_mae": float(history_df["mae"].iloc[-1]) if "mae" in history_df else None,
        "final_val_loss": float(history_df["val_loss"].iloc[-1]) if "val_loss" in history_df else None,
        "final_val_mae": float(history_df["val_mae"].iloc[-1]) if "val_mae" in history_df else None,
        "epochs_ran": int(len(history_df)),
        "best_val_loss": float(history_df["val_loss"].min()) if "val_loss" in history_df else None,
        "best_val_mae": float(history_df["val_mae"].min()) if "val_mae" in history_df else None,
    }

    return {
        "config": {
            "source": "history_fallback",
            "model_file": model_choice.name,
            "model_modified": model_choice.stat().st_mtime,
        },
        "metrics": metrics,
    }


def show_run_metadata(registry_data: dict, run_name: str, model_choice: Path, logs_dir: Path):
    run_meta = resolve_run_metadata(registry_data, run_name, model_choice)
    if not run_meta:
        run_meta = build_history_fallback(run_name, logs_dir, model_choice)
        if run_meta:
            st.info("⚠️ Registry metadata not found. Showing metrics from training CSV fallback.")
        else:
            st.info("📍 No registry metadata or history CSV found for selected model.")
            return

    cfg = run_meta.get("config", {})
    metrics = run_meta.get("metrics", {})
    pipeline_type = infer_pipeline_type(run_name, run_meta)

    # Backfill best validation metrics from history when registry doesn't include them.
    if metrics.get("best_val_loss") is None or metrics.get("best_val_mae") is None:
        history_df = load_history(run_name, logs_dir)
        if history_df is not None:
            if metrics.get("best_val_loss") is None and "val_loss" in history_df.columns:
                metrics["best_val_loss"] = float(history_df["val_loss"].min())
            if metrics.get("best_val_mae") is None and "val_mae" in history_df.columns:
                metrics["best_val_mae"] = float(history_df["val_mae"].min())

    left, right = st.columns(2)
    with left:
        pipeline_chip = chip_html(pipeline_type.upper(), "accent2" if pipeline_type == "sharp" else "brand")
        st.markdown(
            f'<div class="image-frame-label">⚙️&nbsp; Training Config &nbsp;{pipeline_chip}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            config_panel_html({
                "source": cfg.get("source", "registry"),
                "batch_size": cfg.get("batch_size"),
                "epochs": cfg.get("epochs"),
                "learning_rate": cfg.get("learning_rate"),
                "loss_type": cfg.get("loss_type"),
                "ssim_weight": cfg.get("ssim_weight"),
                "augmentation": cfg.get("augmentation_profile"),
                "model_file": cfg.get("model_file", model_choice.name),
            }),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="image-frame-label">📊&nbsp; Final Metrics</div>', unsafe_allow_html=True)
        raw_metrics = {
            "final_loss": metrics.get("final_loss"),
            "final_mae": metrics.get("final_mae"),
            "final_val_loss": metrics.get("final_val_loss"),
            "final_val_mae": metrics.get("final_val_mae"),
            "epochs_ran": metrics.get("epochs_ran"),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_val_mae": metrics.get("best_val_mae"),
        }
        st.markdown(config_panel_html(raw_metrics), unsafe_allow_html=True)

        # Mini training history sparkline if available
        history_df = load_history(run_name, logs_dir)
        if history_df is not None and "val_loss" in history_df.columns and _PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df.get("epoch", list(range(len(history_df)))),
                y=history_df["val_loss"],
                mode="lines",
                line=dict(color="#00c8ff", width=2),
                name="val_loss",
                fill="tozeroy",
                fillcolor="rgba(0,200,255,0.07)",
            ))
            if "val_mae" in history_df.columns:
                fig.add_trace(go.Scatter(
                    x=history_df.get("epoch", list(range(len(history_df)))),
                    y=history_df["val_mae"],
                    mode="lines",
                    line=dict(color="#a78bfa", width=2, dash="dot"),
                    name="val_mae",
                ))
            fig.update_layout(**DEEP_PLOTLY_LAYOUT, height=200, title="Validation History",
                              title_font=dict(family="Orbitron,sans-serif", size=11, color="#00c8ff"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def load_history(run_name: str, logs_dir: Path):
    history_path = logs_dir / f"{run_name}_training.csv"
    if not history_path.exists():
        return None
    try:
        return pd.read_csv(history_path)
    except Exception:
        return None


def run_inference_view(model_choice: Path, img_size: int, registry_data: dict, logs_dir: Path):
    run_name = get_run_name_from_model_path(model_choice)
    st.markdown(
        section_header_html("🖼️", "Image Enhancement", "Tune output detail, upload an image, and run the enhancement pipeline."),
        unsafe_allow_html=True,
    )

    control_col, upload_col = st.columns([1.1, 1.4], gap="large")

    with control_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="image-frame-label">🎛️&nbsp; Output Controls</div>', unsafe_allow_html=True)
        enable_detail_fusion = st.checkbox("Recover fine detail from original", value=True, key="infer_detail_fusion")
        detail_strength = st.slider("Detail strength", min_value=0.0, max_value=1.0, value=0.45, step=0.05, key="infer_detail_strength")
        enable_sharpen = st.checkbox("Apply mild sharpening", value=True, key="infer_sharpen")
        sharpen_amount = st.slider("Sharpen amount", min_value=0.0, max_value=1.5, value=0.40, step=0.05, key="infer_sharpen_amount")
        st.markdown("</div>", unsafe_allow_html=True)
        enhance_clicked = st.button("⚡ Enhance Image", type="primary", key="infer_enhance_btn")

    uploaded_file = None
    pil_image = None
    with upload_col:
        st.markdown('<div class="image-frame-label">📂&nbsp; Input Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload image (JPG, PNG, BMP — max 20 MB)",
            type=["jpg", "jpeg", "png", "bmp"],
            key="infer_uploader",
        )

        if uploaded_file is not None:
            if uploaded_file.size > 20 * 1024 * 1024:
                st.error("Image file is too large (max 20 MB).")
                return

            # Fix V5: Use Pillow's UnidentifiedImageError for robust format validation.
            uploaded_file.seek(0)
            try:
                probe = Image.open(uploaded_file)
                probe.verify()
            except UnidentifiedImageError:
                st.error("Uploaded file is not a recognised image (expected JPG/PNG/BMP).")
                return
            except Exception:
                st.error("Could not read uploaded file — it may be corrupt.")
                return
            finally:
                uploaded_file.seek(0)

            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, use_container_width=True, caption="Input preview")

    if uploaded_file is None or pil_image is None:
        st.markdown(
            empty_state_html("🌊", "No Image Uploaded", "Select a model from the sidebar and upload an image to begin enhancement."),
            unsafe_allow_html=True,
        )
        st.markdown(section_header_html("📋", "Model Metadata"), unsafe_allow_html=True)
        show_run_metadata(registry_data, run_name, model_choice, logs_dir)
        return

    # Image metadata card shown immediately after upload
    h_orig, w_orig = np.array(pil_image).shape[:2]
    ch = np.array(pil_image).shape[2] if len(np.array(pil_image).shape) == 3 else 1
    size_kb = uploaded_file.size / 1024
    fmt = Path(uploaded_file.name).suffix.lstrip(".").lower()
    with upload_col:
        st.markdown(
            image_meta_card_html(w_orig, h_orig, ch, size_kb, fmt),
            unsafe_allow_html=True,
        )

    # Resolve authoritative input size from the model weights (not filename heuristic).
    # @st.cache_resource means this load is essentially free on repeated calls.
    try:
        _m = load_model(str(model_choice))
        img_size = int(_m.input_shape[1])
    except Exception:
        pass  # fall back to heuristic img_size already set by caller

    original_arr, model_input, preprocess_meta = preprocess_image(pil_image, img_size)

    if enhance_clicked:
        t_start = time.perf_counter()
        with st.spinner("Running neural enhancement..."):
            model = load_model(str(model_choice))
            prediction = run_inference(model, model_input)
            h, w = original_arr.shape[:2]
            enhanced = postprocess_prediction(prediction, preprocess_meta, (w, h))

            if enable_detail_fusion:
                enhanced = fuse_details_from_original(
                    original_rgb=original_arr,
                    enhanced_rgb=enhanced,
                    detail_strength=detail_strength,
                    sharpen_amount=(sharpen_amount if enable_sharpen else 0.0),
                )
            elif enable_sharpen and sharpen_amount > 0.0:
                enhanced_blur = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
                enhanced = cv2.addWeighted(
                    enhanced.astype(np.float32),
                    1.0 + sharpen_amount,
                    enhanced_blur,
                    -sharpen_amount,
                    0,
                )
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Feature 3 — PSNR / SSIM quality metrics
        metrics_q = compute_quality_metrics(original_arr, enhanced)

        # Feature 5 — Log to session analytics
        _log_inference(model_choice.name, img_size, elapsed_ms, metrics_q.get("psnr"))

        st.markdown(timing_badge_html(elapsed_ms), unsafe_allow_html=True)

        st.markdown(
            section_header_html("📊", "Quality Metrics", "Objective scores: PSNR, SSIM, brightness & contrast vs original."),
            unsafe_allow_html=True,
        )
        st.markdown(quality_cards_html(metrics_q), unsafe_allow_html=True)

        # Feature 1 — Interactive drag comparison slider
        st.markdown(
            section_header_html("⇔", "Interactive Comparison", "Drag the divider left/right to compare original vs enhanced."),
            unsafe_allow_html=True,
        )
        safe_name = safe_stem(uploaded_file.name)
        image_comparison_slider(
            original=original_arr,
            enhanced=enhanced,
            label_left="Original",
            label_right="Enhanced",
            height=420,
            component_key=f"infer_{safe_name}",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(image_frame_html("🔵  Original"), unsafe_allow_html=True)
            st.image(original_arr, use_container_width=True)
        with col2:
            st.markdown(image_frame_html("✨  Enhanced"), unsafe_allow_html=True)
            st.image(enhanced, use_container_width=True)

        st.download_button(
            label="⬇️  Download Enhanced PNG",
            data=image_to_download_bytes(enhanced),
            file_name=f"enhanced_{safe_name}.png",
            mime="image/png",
            key="infer_download",
        )

    st.markdown(section_header_html("📋", "Model Metadata"), unsafe_allow_html=True)
    show_run_metadata(registry_data, run_name, model_choice, logs_dir)


def run_video_view(model_choice: Path, img_size: int):
    st.markdown(
        section_header_html("🎬", "Video Enhancement", "Upload a video, process it with the selected model, and download the enhanced output."),
        unsafe_allow_html=True,
    )

    preset = st.selectbox(
        "Quality preset",
        options=["Fast", "Balanced", "Sharp", "Custom"],
        index=2,
        key="video_quality_preset",
    )

    preset_values = {
        "Fast": {"enable_detail_fusion": False, "detail_strength": 0.0, "sharpen_amount": 0.0},
        "Balanced": {"enable_detail_fusion": True, "detail_strength": 0.45, "sharpen_amount": 0.30},
        "Sharp": {"enable_detail_fusion": True, "detail_strength": 0.60, "sharpen_amount": 0.45},
    }

    if preset == "Custom":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            enable_detail_fusion = st.checkbox("Recover detail from original", value=True, key="video_detail_fusion")
        with col_b:
            detail_strength = st.slider(
                "Detail strength",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
                key="video_detail_strength",
            )
        with col_c:
            sharpen_amount = st.slider(
                "Sharpen amount",
                min_value=0.0,
                max_value=1.5,
                value=0.45,
                step=0.05,
                key="video_sharpen_amount",
            )
    else:
        selected = preset_values[preset]
        enable_detail_fusion = selected["enable_detail_fusion"]
        detail_strength = selected["detail_strength"]
        sharpen_amount = selected["sharpen_amount"]
        st.caption(
            f"Preset settings -> detail recovery: {enable_detail_fusion}, "
            f"detail strength: {detail_strength:.2f}, sharpen: {sharpen_amount:.2f}"
        )

    max_frames = st.slider(
        "Max frames to process (0 = all)",
        min_value=0,
        max_value=3000,
        value=0,
        step=50,
        key="video_max_frames",
    )

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        if uploaded_video.size > 200 * 1024 * 1024:
            st.error("Video file is too large (max 200MB limit).")
            return
            
        header = uploaded_video.read(16)
        uploaded_video.seek(0)
        is_mp4 = b'ftyp' in header
        is_mkv = b'\x1aE\xdf\xa3' in header
        is_avi = b'RIFF' in header
        # mov is typically ftyp as well or mdat
        if not (is_mp4 or is_mkv or is_avi or b'mdat' in header):
            st.warning("Video signature may be invalid or unsupported. Proceeding with caution.")

    if uploaded_video is None:
        st.markdown(
            empty_state_html("🎞️", "No Video Uploaded", "Upload a video file to begin frame-by-frame enhancement."),
            unsafe_allow_html=True,
        )
        return

    with st.expander("📹 Preview uploaded video (optional)", expanded=False):
        st.video(uploaded_video)

    if not st.button("⚡ Enhance Video", type="primary", key="enhance_video_button"):
        return

    results_dir = Path("results/processed_videos")
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_video.name).suffix if Path(uploaded_video.name).suffix else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(uploaded_video.getbuffer())
        input_path = Path(temp_input.name)

    safe_model_name = model_choice.stem.replace(" ", "_")
    output_path = results_dir / f"{Path(uploaded_video.name).stem}_enhanced_{safe_model_name}.mp4"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_target = total_frames if max_frames == 0 else min(total_frames, max_frames)
    if frames_target <= 0:
        st.error("Video has no readable frames.")
        cap.release()
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    # ETA estimation strip
    secs_per_frame_est = 0.25  # conservative estimate for display
    eta_secs = int(frames_target * secs_per_frame_est)
    eta_label = f"{eta_secs//60}m {eta_secs%60}s" if eta_secs >= 60 else f"~{eta_secs}s"
    st.markdown(
        f"""
        <div class="eta-strip">
          <span>📽️ <strong style="color:#e2eaf5">{width}×{height}</strong> &nbsp;&nbsp;
          ⏱️ FPS <span class="eta-value">{fps:.1f}</span> &nbsp;&nbsp;
          🖼Frames: <span class="eta-value">{frames_target}</span>&nbsp;&nbsp;
          ⏳ Estimated time: <span class="eta-value">{eta_label}</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    model = load_model(str(model_choice))

    progress = st.progress(0)
    status = st.empty()

    processed = 0
    while processed < frames_target:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_rgb, model_input, preprocess_meta = preprocess_rgb_array(frame_rgb, img_size)

        prediction = run_inference(model, model_input)
        enhanced_rgb = postprocess_prediction(prediction, preprocess_meta, (width, height))

        if enable_detail_fusion:
            enhanced_rgb = fuse_details_from_original(
                original_rgb=original_rgb,
                enhanced_rgb=enhanced_rgb,
                detail_strength=detail_strength,
                sharpen_amount=sharpen_amount,
            )

        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        writer.write(enhanced_bgr)

        processed += 1
        progress.progress(int((processed / frames_target) * 100))
        status.markdown(
            f'<span style="font-family:\'Orbitron\',sans-serif;font-size:0.8rem;color:#00c8ff">'
            f'Processing {processed}/{frames_target} frames...</span>',
            unsafe_allow_html=True,
        )

    cap.release()
    writer.release()
    try:
        input_path.unlink(missing_ok=True)
    except Exception:
        pass

    if processed == 0:
        st.error("No frames were processed.")
        return

    st.success(f"✅ Enhancement complete — {processed} frames processed.")
    preview_path, preview_warning = make_web_preview_video(output_path)
    if preview_warning:
        st.warning(preview_warning)

    # Fix N10: stream from disk — no RAM buffering
    st.video(str(preview_path))

    with open(output_path, "rb") as f:
        st.download_button(
            label="⬇️  Download Enhanced Video",
            data=f,
            file_name=safe_stem(output_path.name) + ".mp4",
            mime="video/mp4",
            key="download_enhanced_video",
        )


def run_threat_detection_view(model_choice: Path):
    st.markdown(
        section_header_html("🎯", "Threat Detection", "Run YOLO-powered underwater threat detection on image or video. Enhancement is optional."),
        unsafe_allow_html=True,
    )

    yolo_options = list_yolo_model_files()
    yolo_choice = st.selectbox("YOLO model", options=yolo_options, index=0, key="threat_yolo_model")

    profiles = {
        "Recall": {"conf": 0.10, "enhance": False},
        "Balanced": {"conf": 0.20, "enhance": False},
        "Strict (Recommended)": {"conf": 0.30, "enhance": False},
        "Enhancement Fallback": {"conf": 0.20, "enhance": True},
    }

    if "threat_conf" not in st.session_state:
        st.session_state["threat_conf"] = profiles["Strict (Recommended)"]["conf"]
    if "threat_use_enhancement" not in st.session_state:
        st.session_state["threat_use_enhancement"] = profiles["Strict (Recommended)"]["enhance"]

    profile_name = st.selectbox(
        "Detection profile",
        options=list(profiles.keys()),
        index=2,
        key="threat_profile",
        help="Profiles prefill confidence and enhancement settings based on validation sweeps.",
    )

    if st.session_state.get("threat_profile_applied") != profile_name:
        st.session_state["threat_conf"] = profiles[profile_name]["conf"]
        st.session_state["threat_use_enhancement"] = profiles[profile_name]["enhance"]
        st.session_state["threat_profile_applied"] = profile_name

    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, step=0.05, key="threat_conf")
    use_enhancement = st.checkbox("Apply enhancement before YOLO", key="threat_use_enhancement")

    detector = load_threat_detector(str(model_choice), yolo_choice, float(conf), bool(use_enhancement))
    mode = st.radio("Threat mode", options=["Image", "Video"], horizontal=True, key="threat_mode")

    if mode == "Image":
        uploaded_img = st.file_uploader(
            "Upload image for threat detection",
            type=["jpg", "jpeg", "png", "bmp"],
            key="threat_image_uploader",
        )

        if uploaded_img is None:
            st.markdown(
                empty_state_html("🎯", "No Image Uploaded", "Upload an image to run YOLO threat detection."),
                unsafe_allow_html=True,
            )
            return

        img = np.array(Image.open(uploaded_img).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if st.button("🔍 Detect Threats", type="primary", key="threat_image_btn"):
            with st.spinner("Running threat detection..."):
                detections, annotated_bgr = detector.detect_threats(img_bgr)

            st.markdown(detection_badge_html(len(detections)), unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(image_frame_html("🔵  Input"), unsafe_allow_html=True)
                st.image(img, use_container_width=True)
            with col2:
                st.markdown(image_frame_html("🎯  Detection Output"), unsafe_allow_html=True)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            if detections:
                df_det = pd.DataFrame(detections)
                st.dataframe(
                    df_det,
                    use_container_width=True,
                    column_config={
                        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                        "threat_level": st.column_config.TextColumn("Threat Level"),
                    },
                )
                # Plotly confidence histogram
                if _PLOTLY_AVAILABLE and "confidence" in df_det.columns:
                    fig_hist = go.Figure(go.Histogram(
                        x=df_det["confidence"],
                        nbinsx=20,
                        marker_color="#00c8ff",
                        marker_line_color="#081428",
                        marker_line_width=1,
                        opacity=0.8,
                    ))
                    fig_hist.update_layout(**DEEP_PLOTLY_LAYOUT, height=200,
                                          xaxis_title="Confidence", yaxis_title="Count",
                                          title="Confidence Distribution",
                                          title_font=dict(family="Orbitron,sans-serif",size=11,color="#00c8ff"))
                    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

                # CSV export for detections
                csv_bytes = df_det.to_csv(index=False).encode("utf-8")
                safe_img_stem = safe_stem(uploaded_img.name)
                col_img, col_csv = st.columns(2)
                with col_img:
                    st.download_button(
                        label="⬇️  Download Detected Image",
                        data=image_to_download_bytes(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)),
                        file_name=f"detected_{safe_img_stem}.png",
                        mime="image/png",
                        key="threat_image_download",
                    )
                with col_csv:
                    st.download_button(
                        label="📊  Export Detections CSV",
                        data=csv_bytes,
                        file_name=f"detections_{safe_img_stem}.csv",
                        mime="text/csv",
                        key="threat_image_csv",
                    )
            else:
                safe_img_stem = safe_stem(uploaded_img.name)
                st.download_button(
                    label="⬇️  Download Detected Image",
                    data=image_to_download_bytes(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)),
                    file_name=f"detected_{safe_img_stem}.png",
                    mime="image/png",
                    key="threat_image_download_clean",
                )

        return

    uploaded_video = st.file_uploader(
        "Upload video for threat detection",
        type=["mp4", "avi", "mov", "mkv"],
        key="threat_video_uploader",
    )
    if uploaded_video is None:
        st.markdown(
            empty_state_html("🎬", "No Video Uploaded", "Upload a video to run per-frame YOLO threat detection."),
            unsafe_allow_html=True,
        )
        return

    with st.expander("📹 Preview uploaded video (optional)", expanded=False):
        st.video(uploaded_video)
    max_frames = st.slider(
        "Max frames to process (0 = all)",
        min_value=0,
        max_value=3000,
        value=0,
        step=50,
        key="threat_video_max_frames",
    )

    if not st.button("🔍 Detect Threats (Video)", type="primary", key="threat_video_btn"):
        return

    out_dir = Path("results/detections")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_video.name).suffix if Path(uploaded_video.name).suffix else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(uploaded_video.getbuffer())
        input_path = Path(temp_input.name)

    output_path = out_dir / f"{Path(uploaded_video.name).stem}_detected.mp4"
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_target = total if max_frames == 0 else min(total, max_frames)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    progress = st.progress(0)
    status = st.empty()

    all_detections = []
    processed = 0
    while processed < frames_target:
        ok, frame = cap.read()
        if not ok:
            break

        detections, annotated = detector.detect_threats(frame)
        writer.write(annotated)
        all_detections.extend(detections)

        processed += 1
        progress.progress(int((processed / max(1, frames_target)) * 100))
        status.markdown(
            f'<span style="font-family:\'Orbitron\',sans-serif;font-size:0.8rem;color:#00c8ff">'
            f'Processing {processed}/{frames_target} frames...</span>',
            unsafe_allow_html=True,
        )

    cap.release()
    writer.release()
    try:
        input_path.unlink(missing_ok=True)
    except Exception:
        pass

    st.markdown(detection_badge_html(len(all_detections)), unsafe_allow_html=True)
    st.success(f"✅ Threat detection complete — {processed} frames | {len(all_detections)} detections total.")
    preview_path, preview_warning = make_web_preview_video(output_path)
    if preview_warning:
        st.warning(preview_warning)
    # Fix N10: stream from disk — no RAM buffering
    st.video(str(preview_path))

    safe_vid_stem = safe_stem(output_path.stem)
    col_dl, col_csv2 = st.columns(2)
    with col_dl:
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️  Download Detected Video",
                data=f,
                file_name=f"{safe_vid_stem}.mp4",
                mime="video/mp4",
                key="threat_video_download",
            )
    with col_csv2:
        if all_detections:
            csv_vid = pd.DataFrame(all_detections).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📊  Export Detections CSV",
                data=csv_vid,
                file_name=f"detections_{safe_vid_stem}.csv",
                mime="text/csv",
                key="threat_video_csv",
            )


def run_comparison_view(registry_data: dict, models, logs_dir: Path):
    st.markdown(
        section_header_html("📊", "Run Comparison", "Compare validation curves and final metrics side-by-side."),
        unsafe_allow_html=True,
    )

    run_names = sorted({*registry_data.keys(), *[get_run_name_from_model_path(p) for p in models]})
    if len(run_names) < 2:
        st.markdown(
            empty_state_html("📊", "Not Enough Runs", "At least two registered training runs are needed to compare."),
            unsafe_allow_html=True,
        )
        return

    col_a, col_b = st.columns(2)
    with col_a:
        run_a = st.selectbox("Run A", run_names, index=max(len(run_names) - 2, 0))
    with col_b:
        run_b = st.selectbox("Run B", run_names, index=max(len(run_names) - 1, 0))

    if run_a == run_b:
        st.warning("Choose two different runs.")
        return

    metrics_a = registry_data.get(run_a, {}).get("metrics", {})
    metrics_b = registry_data.get(run_b, {}).get("metrics", {})
    st.caption(
        f"Run A pipeline: **{infer_pipeline_type(run_a, registry_data.get(run_a))}** | "
        f"Run B pipeline: **{infer_pipeline_type(run_b, registry_data.get(run_b))}**"
    )

    rows = []
    for metric in ["final_loss", "final_mae", "final_val_loss", "final_val_mae", "epochs_ran"]:
        a = metrics_a.get(metric)
        b = metrics_b.get(metric)
        delta = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        rows.append({"metric": metric, "run_a": a, "run_b": b, "delta_b_minus_a": delta})

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        column_config={
            "delta_b_minus_a": st.column_config.NumberColumn("Δ (B − A)", format="%.6f"),
            "run_a": st.column_config.NumberColumn(f"Run A: {run_a}", format="%.6f"),
            "run_b": st.column_config.NumberColumn(f"Run B: {run_b}", format="%.6f"),
        },
    )

    hist_a = load_history(run_a, logs_dir)
    hist_b = load_history(run_b, logs_dir)
    if hist_a is None or hist_b is None:
        st.info("History CSV missing for one or both runs. Metric table is still available.")
        return

    required_cols = {"epoch", "val_loss", "val_mae"}
    if not required_cols.issubset(set(hist_a.columns)) or not required_cols.issubset(set(hist_b.columns)):
        st.info("Validation columns missing in one or both history CSV files.")
        return

    # Join by epoch to safely handle runs with different lengths.
    hist_a_view = hist_a[["epoch", "val_loss", "val_mae"]].copy()
    hist_b_view = hist_b[["epoch", "val_loss", "val_mae"]].copy()
    hist_a_view.columns = ["epoch", f"{run_a}_val_loss", f"{run_a}_val_mae"]
    hist_b_view.columns = ["epoch", f"{run_b}_val_loss", f"{run_b}_val_mae"]

    chart_df = pd.merge(hist_a_view, hist_b_view, on="epoch", how="outer").sort_values("epoch")
    st.markdown(section_header_html("📈", "Validation Curves"), unsafe_allow_html=True)

    if _PLOTLY_AVAILABLE:
        PALETTE = ["#00c8ff", "#a78bfa", "#10b981", "#f5c518"]
        fig = go.Figure()
        for i, col in enumerate([c for c in chart_df.columns if "val_loss" in c]):
            fig.add_trace(go.Scatter(
                x=chart_df["epoch"], y=chart_df[col],
                mode="lines+markers", name=col,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
                marker=dict(size=4),
            ))
        for i, col in enumerate([c for c in chart_df.columns if "val_mae" in c]):
            fig.add_trace(go.Scatter(
                x=chart_df["epoch"], y=chart_df[col],
                mode="lines", name=col,
                line=dict(color=PALETTE[(i + 2) % len(PALETTE)], width=2, dash="dot"),
            ))
        fig.update_layout(**DEEP_PLOTLY_LAYOUT, height=370,
                          xaxis_title="Epoch", yaxis_title="Metric Value",
                          title="Validation Loss & MAE over Epochs",
                          title_font=dict(family="Orbitron,sans-serif", size=13, color="#00c8ff"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(chart_df.set_index("epoch"))


def run_recommender_view(registry_data: dict, models, logs_dir: Path):
    st.markdown(
        section_header_html("🏆", "Best Run Recommender", "Rank all training runs by a chosen metric to find the top checkpoint."),
        unsafe_allow_html=True,
    )

    run_names = sorted({*registry_data.keys(), *[get_run_name_from_model_path(p) for p in models]})
    if not run_names:
        st.markdown(
            empty_state_html("🏆", "No Runs Available", "No registered training runs found."),
            unsafe_allow_html=True,
        )
        return

    metric_options = {
        "Best val_loss (from history CSV)": "best_val_loss",
        "Best val_mae (from history CSV)": "best_val_mae",
        "Final val_loss (from registry)": "final_val_loss",
        "Final val_mae (from registry)": "final_val_mae",
        "Final loss (from registry)": "final_loss",
        "Final mae (from registry)": "final_mae",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        metric_label = st.selectbox("Rank by metric", list(metric_options.keys()))
    with col2:
        top_k = st.slider("Top K", min_value=3, max_value=20, value=8, step=1)

    metric_key = metric_options[metric_label]
    ranking_rows = []

    for run_name in run_names:
        run_cfg = registry_data.get(run_name, {}).get("config", {})
        run_metrics = registry_data.get(run_name, {}).get("metrics", {})

        metric_value = None
        if metric_key in {"best_val_loss", "best_val_mae"}:
            history_df = load_history(run_name, logs_dir)
            if history_df is not None:
                target_col = "val_loss" if metric_key == "best_val_loss" else "val_mae"
                if target_col in history_df.columns and len(history_df[target_col].dropna()) > 0:
                    metric_value = float(history_df[target_col].min())
        else:
            raw_value = run_metrics.get(metric_key)
            if isinstance(raw_value, (int, float)):
                metric_value = float(raw_value)

        if metric_value is None:
            continue

        ranking_rows.append(
            {
                "run_name": run_name,
                "pipeline_type": infer_pipeline_type(run_name, registry_data.get(run_name)),
                "score": metric_value,
                "epochs": run_metrics.get("epochs_ran"),
                "batch_size": run_cfg.get("batch_size"),
                "learning_rate": run_cfg.get("learning_rate"),
                "augmentation_profile": run_cfg.get("augmentation_profile"),
            }
        )

    if not ranking_rows:
        st.warning("No runs had enough data to rank for the selected metric.")
        return

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values("score", ascending=True).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    best_row = ranking_df.iloc[0]
    st.markdown(
        winner_card_html(best_row["run_name"], best_row["score"], metric_key),
        unsafe_allow_html=True,
    )

    top_df = ranking_df.head(top_k)
    if _PLOTLY_AVAILABLE:
        PALETTE = ["#00c8ff","#a78bfa","#10b981","#f5c518","#ff4d6a","#38bdf8","#fb923c","#84cc16"]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(top_df))]
        fig_bar = go.Figure(go.Bar(
            y=top_df["run_name"],
            x=top_df["score"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#081428", width=1)),
            text=[f"{s:.5f}" for s in top_df["score"]],
            textposition="inside",
            textfont=dict(family="Orbitron,sans-serif", size=10, color="#081428"),
        ))
        fig_bar.update_layout(**DEEP_PLOTLY_LAYOUT, height=max(280, top_k * 32),
                              xaxis_title=metric_key, yaxis_title="",
                              yaxis=dict(**DEEP_PLOTLY_LAYOUT["yaxis"], autorange="reversed"),
                              title=f"Top {top_k} Runs — {metric_key}",
                              title_font=dict(family="Orbitron,sans-serif", size=13, color="#00c8ff"))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        top_df,
        use_container_width=True,
        column_config={
            "rank": st.column_config.NumberColumn("🥇 Rank", format="%d"),
            "score": st.column_config.ProgressColumn(
                "Score", min_value=0.0,
                max_value=float(ranking_df["score"].max()) * 1.1,
                format="%.6f",
            ),
        },
    )


def main():
    if "health" in st.query_params:
        st.json({"status": "ok", "version": APP_VERSION})
        st.stop()

    apply_global_styles()
    render_hero()

    device_info = configure_tensorflow_device({})

    try:
        runtime_cfg = load_runtime_config()
    except Exception as exc:
        st.error(f"Failed to load config.yaml: {exc}")
        st.stop()

    checkpoint_dir = Path(runtime_cfg.get("checkpoint_dir", "models/checkpoints"))
    registry_path = Path(runtime_cfg.get("registry_path", "results/model_registry.json"))
    logs_dir = Path("logs/csv")

    if not checkpoint_dir.exists():
        st.error(f"Checkpoint directory not found: {_e(str(checkpoint_dir))}")
        st.stop()

    models = list_model_files(checkpoint_dir)
    if not models:
        st.warning("No model files found in the checkpoint directory.")
        st.stop()

    registry_data = load_registry(registry_path)

    # ── Stat cards ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(stat_card_html("🧠", len(models), "Models Available"), unsafe_allow_html=True)
    with c2:
        st.markdown(stat_card_html("📋", len(registry_data), "Registry Runs"), unsafe_allow_html=True)
    with c3:
        gpu_count = device_info.get("gpu_count", 0)
        st.markdown(stat_card_html("⚡", gpu_count, "GPUs Detected"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
              <span class="sidebar-logo-icon">🌊</span>
              <span class="sidebar-logo-name">AquaIntel Vision</span>
            </div>
            <hr class="sidebar-divider">
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🧠 Model Checkpoint", expanded=True):
            default_index = 0
            preferred = [
                i for i, p in enumerate(models)
                if ("256" in p.stem and p.suffix == ".keras" and "_final" in p.stem)
            ]
            if preferred:
                default_index = preferred[0]

            model_choice = st.selectbox(
                "Checkpoint",
                options=models,
                index=default_index,
                format_func=lambda p: p.name,
                key="sidebar_model_choice",
            )

            # Fix N6: Path.is_relative_to() prevents path-traversal
            try:
                resolved = model_choice.resolve()
                resolved_dir = checkpoint_dir.resolve()
                if not resolved.is_relative_to(resolved_dir):
                    st.error(
                        f"Security: model path '{_e(model_choice.name)}' escapes the "
                        "allowed checkpoint directory."
                    )
                    st.stop()
            except Exception as exc:
                st.error(f"Could not resolve model path: {_e(str(exc))}")
                st.stop()

            img_size = get_model_input_size(str(model_choice))
            updated_at = datetime.fromtimestamp(model_choice.stat().st_mtime)

            st.markdown(
                f'Input size: {chip_html(f"{img_size}×{img_size} px", "brand")}<br>'
                f'<span style="font-size:0.75rem;color:#6b8aad">Updated: {updated_at:%Y-%m-%d %H:%M}</span>',
                unsafe_allow_html=True,
            )

        with st.expander("⚡ Runtime Info", expanded=False):
            st.markdown(
                f"""
                <table class="meta-table">
                  <tr><td>Device</td><td>{_e(str(device_info.get("device", "—")))}</td></tr>
                  <tr><td>GPU count</td><td>{_e(str(device_info.get("gpu_count", 0)))}</td></tr>
                  <tr><td>Mixed precision</td><td>{"Yes" if device_info.get("mixed_precision") else "No"}</td></tr>
                </table>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📁 Registry Stats", expanded=False):
            st.markdown(
                f"""
                <table class="meta-table">
                  <tr><td>Runs</td><td>{_e(str(len(registry_data)))}</td></tr>
                  <tr><td>Checkpoint dir</td><td>{_e(str(checkpoint_dir))}</td></tr>
                  <tr><td>Registry path</td><td>{_e(str(registry_path))}</td></tr>
                </table>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🚀 Quick Start", expanded=False):
            st.markdown(
                """
                1. 🧠 **Pick a checkpoint** above  
                2. 📂 **Open a tab** below  
                3. ⬆️ **Upload** an image or video  
                4. ⚡ **Click** the action button  
                5. ⬇️ **Download** your result
                """,
                unsafe_allow_html=True,
            )

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_infer, tab_batch, tab_video, tab_threat, tab_compare, tab_recommend, tab_live = st.tabs([
        "🖼 Inference",
        "📦 Batch",
        "🎬 Video",
        "🎯 Threat Detection",
        "📊 Run Comparison",
        "🏆 Best Run",
        "📡 Live Training",
    ])

    # Track tab visits
    _init_analytics()

    with tab_infer:
        st.session_state["analytics"]["tabs_visited"].add("inference")
        run_inference_view(model_choice=model_choice, img_size=img_size, registry_data=registry_data, logs_dir=logs_dir)
    with tab_batch:
        st.session_state["analytics"]["tabs_visited"].add("batch")
        run_batch_inference_view(model_choice=model_choice, img_size=img_size)
    with tab_video:
        st.session_state["analytics"]["tabs_visited"].add("video")
        run_video_view(model_choice=model_choice, img_size=img_size)
    with tab_threat:
        st.session_state["analytics"]["tabs_visited"].add("threat")
        run_threat_detection_view(model_choice=model_choice)
    with tab_compare:
        st.session_state["analytics"]["tabs_visited"].add("comparison")
        run_comparison_view(registry_data=registry_data, models=models, logs_dir=logs_dir)
    with tab_recommend:
        st.session_state["analytics"]["tabs_visited"].add("recommender")
        run_recommender_view(registry_data=registry_data, models=models, logs_dir=logs_dir)
    with tab_live:
        st.session_state["analytics"]["tabs_visited"].add("live_training")
        run_live_training_view(logs_dir=logs_dir)

    # Feature 5 — Analytics expander in sidebar (always rendered)
    with st.sidebar:
        with st.expander("📊 Session Analytics", expanded=False):
            render_analytics_dashboard()

    render_footer()


if __name__ == "__main__":
    main()
