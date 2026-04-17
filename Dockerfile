# Pin to a specific patch to guarantee a reproducible baseline.
# Update this tag intentionally when you want to upgrade Python.
FROM python:3.11.12-slim

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# ── System deps (OpenCV headless) ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (cached layer) ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ── Create a non-root user to avoid container-escape risks ───────────────────
RUN useradd -m -u 1001 appuser \
    && chown -R appuser:appuser $APP_HOME

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

USER appuser

# ── Network ───────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check (polls the ?health query param handled in main()) ────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/?health=1')" \
    || exit 1

# ── Entrypoint — use the canonical app, not the pass-through shim ─────────────
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
