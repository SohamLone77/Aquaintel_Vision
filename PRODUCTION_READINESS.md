# AquaIntel Vision — Production Readiness Checklist

> Status as of 2026-04-14. Items are ordered by priority within each section.
> Legend: 🔴 Blocker · 🟠 High · 🟡 Medium · 🟢 Nice-to-have

---

## 1. 🧠 Model Quality (ML Readiness)

### YOLO Threat Detector
- 🔴 **Complete YOLO annotations** — Only 49% of train labels and 28% of val labels are real
  annotations; the rest are dummy/placeholder boxes (`annotation_progress.json`).
  The model cannot reach production-class mAP until the dataset is fully annotated.
  - Target: 100% real annotations before any production deployment.
  - Tool already available: `auto_label_remaining.py`, `annotate_check.ps1`.
- 🔴 **mAP50 must reach ≥ 0.75 before promotion** — Current best is 0.192 (`underwater_run5`).
  This is the gate defined in `ExperimentTracker.check_yolo_gates()` and must be met.
  - Action: Retrain on fully-annotated dataset with more epochs (50–100).
- 🟠 **Register YOLO runs in the experiment leaderboard** — `ExperimentTracker` supports
  `register_yolo_experiment()` but it is never called from `train_yolo.py`.
  Wire it in the same way U-Net runs are registered in `train_complete.py`.
- 🟠 **Evaluate YOLO with SSIM / PSNR metrics post-enhancement** — Currently only box-loss
  metrics exist. Run a side-by-side evaluation of raw vs enhanced input to confirm
  whether enhancement helps or hurts detection (see `production_detector.compare_modes()`).

### U-Net Enhancement
- 🟠 **Add PSNR and SSIM scoring at validation time** — Only MAE and combined loss are
  tracked. PSNR/SSIM are the standard perceptual metrics for image restoration and are
  required for credible production comparison. Wire into `UnderwaterTrainer.evaluate()`.
- 🟠 **Visually validate 256x256 models** — `unet256_20260405_0655_final` has higher
  numerical loss (expected due to 4× more pixels) but may look better. Run
  `compare_results.py` or `streamlit_app.py` to confirm before discarding.
- 🟡 **Promote a single champion model** — Set `is_promoted=True` in the registry for the
  chosen production model. `streamlit_app.py` and `production_detector.py` should
  auto-select the promoted model instead of relying on filename sorting.

---

## 2. 💻 Code Quality & Architecture

- 🟠 **Delete or finish `detect_threats.py` and `detect_threats_enhanced.py`** — These
  appear to be superseded by `production_detector.py`. Having three detection entry-points
  creates confusion about which is production.
- 🟠 **Delete or migrate `train_yolo_standardized.py`** — Exists alongside `train_yolo.py`
  with unclear ownership. Consolidate into one authoritative YOLO training script.
- 🟠 **Add structured logging throughout** — All modules rely exclusively on `print()`
  statements. Replace with Python's `logging` module so log levels (DEBUG, INFO, WARNING,
  ERROR) can be controlled at runtime without code changes. Critical for any deployment.
- 🟡 **Pin all dependencies to exact versions** — `requirements.txt` uses `>=` lower-bounds
  only. A `requirements.lock` or `uv.lock` file with exact hashes prevents silent
  dependency regressions across machines and CI.
  ```
  pip freeze > requirements.lock
  ```
- 🟡 **Fix `requirements.txt` — remove `jupyter`** — Jupyter is a development dependency,
  not a runtime one. It should not be installed in a production environment.
- 🟡 **Move `DEFAULT_YOLO_PATH` out of source code** — The hard-coded run path in
  `production_detector.py:26` ties the code to a specific training artefact. Read it
  from `config.yaml` or from the model registry's promoted model instead.
- 🟢 **Add `__init__.py` to `utils/` and `scripts/`** — Both are used as packages but lack
  `__init__.py` files, which can cause import failures in strict environments or
  when packaged.

---

## 3. 🧪 Testing

- 🔴 **Test coverage is limited to smoke tests** — `tests/test_all.py` only checks imports
  and model build shape. There are no behavioural tests for data loading, augmentation,
  loss functions, or inference correctness.
- 🟠 **Add a data pipeline test** — Validate that `UnderwaterDataLoader` correctly splits
  data, applies only the configured augmentation profile, and that train/val indices
  never overlap.
- 🟠 **Add a loss function unit test** — The new `_combined_loss` closure in
  `train_complete.py` should have a unit test asserting output range, differentiability,
  and that `ssim_weight` is correctly applied.
- 🟠 **Add a config validation test** — Assert that loading a valid `config.yaml` produces
  all expected keys and that a malformed file raises `ConfigError`.
- 🟡 **Add a `ModelRegistry` round-trip test** — Write a run, reload the registry,
  confirm the record is intact. Already partially covered in the smoke test added
  during the April-14 audit; expand to cover concurrent writes.
- 🟡 **Add a CI pipeline** — Use GitHub Actions (or equivalent) to run `tests/test_all.py`
  on every push. A minimum viable workflow:
  ```yaml
  # .github/workflows/ci.yml
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - run: pip install -r requirements.txt
        - run: python -m pytest tests/ -v
  ```

---

## 4. 📦 Infrastructure & Deployment

- 🔴 **No Docker / containerisation** — The app has no `Dockerfile`. Without a container,
  reproducing the exact runtime on any server (cloud or on-premise) is error-prone.
  A minimal starting point:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8501
  CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501"]
  ```
- 🟠 **Add a `/health` endpoint** — `streamlit_app.py` has no health-check route.
  For any PaaS/Kubernetes deployment a liveness probe is required. Consider adding
  a minimal FastAPI sidecar or a Streamlit custom route.
- 🟠 **RTSP credentials must not be in plain text** — `video_processor.py` and the README
  example show RTSP URLs with `username:password` embedded in the URL string.
  For production, these should come from environment variables or a secrets manager.
- 🟡 **Use environment variables for all paths** — Only `DATA_PATH` is overridable via env
  var. `checkpoint_dir`, `registry_path`, and `results_dir` should also be overridable
  so the app can run with read-only source code and writable mounted volumes.
- 🟡 **Add resource limits to video processing** — Streamlit video enhancement is fully
  synchronous and blocks the entire UI thread. For production, offload to a background
  worker (e.g. `threading.Thread` or `concurrent.futures`) and stream progress to the UI.

---

## 5. 🔒 Security

- 🟠 **Validate uploaded file MIME types server-side** — `streamlit_app.py` restricts
  uploads by file extension only (e.g. `type=["jpg", "png"]`). A malicious user can
  rename any file. Add `python-magic` or `imghdr` to verify actual byte-level MIME.
- 🟠 **Cap uploaded file size** — No maximum upload size is enforced. A 4 GB video upload
  would consume all RAM and crash the server. Enforce a cap (e.g. 500 MB) before
  writing to disk.
- 🟡 **Sanitize output filenames** — Output paths are derived from uploaded filenames
  (e.g. `f"enhanced_{Path(uploaded_file.name).stem}.png"`). Use `pathlib.Path.name`
  plus a sanitizer to prevent directory traversal.

---

## 6. 📊 Monitoring & Observability

- 🟠 **Log every inference with confidence and latency** — Production threat detection
  should write a structured JSON log entry per frame/image (class, confidence,
  latency_ms, timestamp). Currently only the final `detection_report.json` summary exists.
- 🟠 **Alert on CRITICAL/HIGH threats externally** — On-frame text overlay (`THREAT DETECTED`)
  is only visible to whoever is watching the live window. Wire a webhook, email, or
  push notification when `threat_level in {"CRITICAL", "HIGH"}` is detected.
- 🟡 **Add model drift detection** — Track rolling average of detection confidence over
  time. A sustained drop signals distribution shift (e.g. new camera, new water
  turbidity) and should trigger a retraining alert.
- 🟡 **Expose Streamlit metrics** — Use `st.experimental_memo` cache hit/miss counts and
  track inference latency per model in a sidebar or `results/metrics.json` for ops teams.

---

## 7. 📖 Documentation

- 🟠 **Write a deployment guide** — The README covers local development well but has no
  section on how to deploy to a server, cloud VM, or container registry.
- 🟡 **Document the data schema** — Record the expected folder structure, filename
  conventions, and image format requirements for `data/raw` and `data/reference`
  in a `DATA_SCHEMA.md` file.
- 🟡 **Add docstrings to `streamlit_app.py`** — The 953-line Streamlit app has very few
  module/function docstrings. Adding them improves maintainability significantly.
- 🟢 **Archive the completed TODO files** — `SHARPENING_GUIDE_TODO.md`,
  `STREAMLIT_TODO.md`, and `annotation_checklist.md` are fully or partially done.
  Move them to a `docs/archive/` folder so the root is clean for operators.

---

## Summary — Priority Order

| Priority | Count | Action |
|---|---|---|
| 🔴 Blocker (must fix before go-live) | 3 | Annotate dataset, reach YOLO mAP ≥ 0.75, containerise |
| 🟠 High (fix in this sprint) | 13 | Logging, tests, security, monitoring wiring |
| 🟡 Medium (next sprint) | 10 | Dependency pinning, env vars, drift detection, docs |
| 🟢 Nice-to-have | 3 | Archive old TODOs, `__init__.py`, leaderboard UI |
