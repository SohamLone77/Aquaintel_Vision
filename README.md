---
title: AquaIntel Vision
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

<div align="center">

# 🌊 AquaIntel Vision

**Production-grade underwater image enhancement, real-time threat detection, and model analytics — in one sleek dashboard.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-≥1.38-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/SohamLone77/Aquaintel_Vision)

</div>

---

> 📸 **Hero Screenshot**
> ![AquaIntel Vision in action](web_interface.gif)

---

## ✨ What It Does

AquaIntel Vision is a **full-stack MLOps dashboard** for underwater computer vision research and production deployment. Upload a degraded underwater image or video feed, and the platform instantly:

1. **Enhances it** with a trained U-Net deep-learning model (SSIM + MSE loss)
2. **Detects threats** — divers, boats, and foreign objects — using a fine-tuned YOLO11 detector
3. **Measures quality** objectively with PSNR, SSIM, and brightness/contrast delta metrics
4. **Tracks experiments** with a full model registry and comparative analytics

---

## 🚀 Key Features

### 🧠 AI & Model Features
- **U-Net image enhancement** — Custom architecture trained on the UIEB underwater dataset, producing vivid, defogged output from murky underwater frames
- **YOLO11 threat detection** — Fine-tuned on a custom underwater annotation dataset for real-time bounding-box inference across 9+ threat classes
- **Multi-model checkpoint switching** — Pick any trained `.keras` checkpoint from `models/checkpoints` via a sidebar dropdown without restarting the app
- **Automated model recommendation** — The Model Arena tab surfaces the best-performing run from your registry based on composite PSNR/SSIM scoring

### 📊 Quality & Analytics
- **Objective image metrics** — PSNR (dB), SSIM, brightness Δ, and contrast Δ computed per inference using `scikit-image`
- **Interactive drag comparison slider** — A pure-JS side-by-side slider (no external Streamlit component needed) to visually compare original vs. enhanced output
- **Session analytics dashboard** — Real-time Plotly charts of per-inference latency and video-job FPS across the active session

### 🎬 Video & Batch Processing
- **In-app video enhancement** — Upload `.mp4`/`.avi`, and the app processes frame-by-frame and packages a download-ready enhanced video
- **Batch image processing** — Zip upload support: compress a folder of images, upload once, and download the enhanced batch as a ZIP archive
- **Real-time modes** — Webcam, video file, RTSP stream, and batch-folder processing via the headless `video_processor.py` CLI

### 🏗️ Production Hardening
- **Dockerized** — Single-command deployment with a non-root user, OpenCV system deps, and health-check endpoint baked in
- **Security-first HTML rendering** — All user-supplied values are `html.escape`d before injection; no raw-string interpolation into HTML
- **GPU auto-detection** — `utils/gpu.py` selects GPU/CPU at startup and applies memory-growth policy; overridable with env flags
- **Centralized config** — `config.yaml` drives augmentation profiles, model paths, dataset location, and runtime knobs — no magic constants scattered in code

### 🎨 Premium UI
- **Deep-ocean dark design system** — Custom CSS design tokens, glassmorphism panels, `Orbitron` + `Inter` typography, and animated glow micro-interactions
- **Fully responsive** — Grid layouts collapse gracefully to single-column on narrow viewports
- **Plotly-themed charts** — All data visualisations share the same dark-ocean colour palette

---

## 🌐 Live Demo

> 🚧 **Demo link goes here once deployed to Hugging Face Spaces or Streamlit Community Cloud.**

| Platform | Link |
|---|---|
| **Hugging Face Spaces** | [![Open in HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/SohamLone77/aquaintel-vision) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io/) ≥ 1.38 |
| **Enhancement Model** | [TensorFlow / Keras](https://tensorflow.org/) 2.x — Custom U-Net |
| **Detection Model** | [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) ≥ 8.2 |
| **Deep Learning (PyTorch)** | [PyTorch](https://pytorch.org/) ≥ 2.0 + TorchVision |
| **Computer Vision** | [OpenCV](https://opencv.org/) 4.x, [Pillow](https://pillow.readthedocs.io/) |
| **Data Science** | [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) |
| **Image Quality** | [scikit-image](https://scikit-image.org/) (PSNR, SSIM) |
| **Visualisation** | [Plotly](https://plotly.com/python/) 5.x, [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/) |
| **Augmentation** | [Albumentations](https://albumentations.ai/) ≥ 1.4 |
| **Model Export** | [ONNX](https://onnx.ai/), [onnxruntime](https://onnxruntime.ai/), [onnxslim](https://github.com/onnx/optimizer) |
| **Containerisation** | [Docker](https://www.docker.com/) (Python 3.11 slim base) |
| **Config** | PyYAML 6.x |

---

## 📦 Installation & Local Setup

### Prerequisites

- Python **3.11** (tested; other 3.x may work)
- `git`
- *(Optional)* NVIDIA GPU with CUDA for accelerated inference

### 1 · Clone the repository

```bash
git clone https://github.com/SohamLone77/Aquaintel_Vision.git
cd Aquaintel_Vision
```

### 2 · Create a virtual environment

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
```

```bash
# macOS / Linux
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### 3 · Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 · Run the app

```bash
streamlit run streamlit_app.py
```

The app will open automatically at **http://localhost:8501**.

---

## 🐳 Docker

For a fully reproducible, production-equivalent environment:

```bash
# Build
docker build -t aquaintel-vision .

# Run
docker run -p 8501:8501 aquaintel-vision
```

Then navigate to **http://localhost:8501**.

The container runs as a **non-root user**, exposes port `8501`, and includes a built-in health-check endpoint.

---

## ⚙️ Configuration

### `config.yaml` — Central control panel

All runtime knobs live in `config.yaml`. Edit this file to change model paths, dataset location, and augmentation behaviour **without touching any Python code**.

```yaml
# config.yaml (excerpt)
model:
  checkpoint_dir: models/checkpoints

data:
  raw_dir: data/raw
  reference_dir: data/reference

augmentation:
  enabled: true
  profile: light          # one of: none | light | standard | strong
  flip_prob: 0.5
  brightness_delta: 0.1
  contrast_lower: 0.8
  contrast_upper: 1.2
```

### Environment variables (GPU & runtime)

You can override GPU behaviour with environment variables — no code edits required:

```powershell
# Windows PowerShell
$env:USE_GPU            = "1"   # 1 = enable GPU, 0 = force CPU
$env:GPU_MEMORY_GROWTH  = "1"   # prevent pre-allocating all VRAM
$env:MIXED_PRECISION    = "1"   # enable float16 mixed precision

streamlit run streamlit_app.py
```

```bash
# macOS / Linux
USE_GPU=1 GPU_MEMORY_GROWTH=1 streamlit run streamlit_app.py
```

### Dataset path override

```powershell
$env:DATA_PATH = "D:\datasets\uieb"
streamlit run streamlit_app.py
```

### GPU setup (Windows)

> **Note:** TensorFlow 2.11+ does **not** support native Windows GPU. Use one of the options below.

**Option A — WSL2 + CUDA (recommended for NVIDIA):**
```bash
wsl
pip install tensorflow[and-cuda]
```

**Option B — Legacy native Windows (TF 2.10 + CUDA 11.2 + cuDNN 8.1):**
```powershell
py -3.10 -m venv .venv310
.\.venv310\Scripts\python.exe -m pip install tensorflow==2.10.* "numpy<2"
```

---

## 🖥️ Usage Guide

### First-time user walkthrough

| Step | What to do |
|---|---|
| **1. Select a model** | Use the **🧠 Model** section in the left sidebar to pick a trained checkpoint |
| **2. Choose a tab** | Navigate using the top tab bar: **Enhance**, **Detect**, **Video**, **Batch**, **Model Arena**, **Analytics** |
| **3. Upload an image** | On the **Enhance** tab, drag-and-drop or browse for a `.jpg`, `.png`, or `.bmp` file |
| **4. Run enhancement** | Click **Enhance Image** and watch the Orbitron-styled metrics appear |
| **5. Compare** | Drag the slider handle left/right on the comparison panel to visually inspect the before/after |
| **6. Download** | Click **⬇ Download Enhanced PNG** to save the result |

### Threat detection

1. Switch to the **Detect** tab
2. Upload your image (enhanced output is auto-carried over if you came from **Enhance**)
3. Adjust the **Confidence threshold** slider in the sidebar
4. Click **Run Detection** — annotated bounding boxes and threat counts appear in real time

### Batch processing

1. Zip a folder of images: `Compress-Archive -Path .\input_images -DestinationPath batch.zip`
2. Switch to the **Batch** tab and upload `batch.zip`
3. Click **Process Batch** — a progress bar tracks each frame
4. Download the results as a ZIP archive

### Model comparison (Model Arena)

1. Go to the **Model Arena** tab
2. Select two experiment runs from the dropdowns
3. The app renders metric deltas, validation loss curves, and automatically crowns a winner 🏆

---

## 🗂️ Project Structure

```
Aquaintel_Vision/
├── streamlit_app.py          # Main dashboard (3 100+ lines)
├── production_detector.py    # UnderwaterThreatDetector class
├── video_processor.py        # Headless video / RTSP / webcam CLI
├── train_unet.py             # U-Net training entrypoint
├── train_complete.py         # Core trainer implementation
├── finetune_yolo.py          # YOLO fine-tuning pipeline
├── config.yaml               # Centralised runtime config
├── requirements.txt          # Pinned dependency ranges
├── Dockerfile                # Production container definition
├── utils/
│   ├── config_loader.py      # Reads config.yaml at startup
│   └── gpu.py                # TF device selection & memory growth
├── training/
│   └── data_loader.py        # UIEB dataset loader
├── models/
│   └── checkpoints/          # Trained .keras checkpoints
├── data/
│   ├── raw/                  # Raw (degraded) input images
│   └── reference/            # Clean reference images
└── results/
    └── model_registry.json   # Run metadata for all experiments
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create a feature branch** — `git checkout -b feat/my-amazing-feature`
3. **Commit your changes** — `git commit -m 'feat: add amazing feature'`
4. **Push** — `git push origin feat/my-amazing-feature`
5. **Open a Pull Request** and describe what you changed

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages and ensure any new code is covered by a test in `tests/`.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full text.

```
MIT License

Copyright (c) 2026 Soham Lone

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

Made with 🌊 by **Soham Lone** · [GitHub](https://github.com/SohamLone77)

*AquaIntel Vision v2026.04*

</div>
