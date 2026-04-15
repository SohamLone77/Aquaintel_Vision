# AI

Underwater image enhancement training workspace based on a U-Net pipeline.

## Two Pipelines

### Standard training path

Production-default scripts:

- `train_unet.py` (main training entrypoint)
- `train_complete.py` (core trainer implementation used by `train_unet.py`)
- `scripts/validate_dataset.py` (required pre-check for dataset integrity)

### Sharpness/resolution path

Experimental scripts:

- `train_sharp.py` (edge-preserving loss experimentation)
- `resume_sharp.py` (fine-tuning existing checkpoints for sharper outputs)
- `sharpen_output.py` (post-processing sharpen variants)
- `compare_results.py` (quality comparison/metrics reporting)
- `quick_test_sharp.py` (quick smoke checks for sharp pipeline)

## Current Status

- ✅ Step 1: Centralized runtime config via `config.yaml`.
- ✅ Step 2: Dataset validation enforced before training starts.
- ✅ Step 3: Deterministic dataset download and extract script.
- ✅ Step 4: Model run registry with metadata (`results/model_registry.json`).
- ✅ Environment: Python 3.11 venv (`.venv311`) with all dependencies.
- ✅ Baseline training: `unet_20260403_0811` (50 epochs, SSIM+MSE combined loss).
- ✅ Loss A/B: `ssim_weight=0.5` confirmed as best — kept as default.
- ✅ LR A/B: `learning_rate=1e-4` confirmed as best — kept as default.
- ✅ Batch A/B: `batch_size=8` confirmed as best — kept as default.
- ✅ Augmentation A/B: `profile=light` confirmed as winner — locked in `config.yaml`.
- 🔜 **Next axis: `img_size=256`** — uncomment `# img_size: 256` in `config.yaml` to run.

Progress tracker: see `IMPLEMENTATION_TODO.md`.

## Environment Setup

Create and install dependencies in a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

### GPU Setup (Windows)

This project now auto-detects GPU at runtime and prints the selected device.

Quick check:

```powershell
.\.venv311\Scripts\python.exe -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If your list is empty on native Windows, use one of these options:

1. WSL2 + CUDA (recommended for NVIDIA):

```powershell
wsl
# inside Ubuntu/WSL:
python -m pip install --upgrade pip
python -m pip install tensorflow[and-cuda]
```

2. Native Windows legacy stack (NVIDIA CUDA, no WSL):

```powershell
# Requires Python 3.10 + TensorFlow 2.10 + CUDA 11.2 + cuDNN 8.1
py -3.10 -m venv .venv310
.\.venv310\Scripts\python.exe -m pip install --upgrade pip
.\.venv310\Scripts\python.exe -m pip install tensorflow==2.10.*
.\.venv310\Scripts\python.exe -m pip install "numpy<2"
.\.venv310\Scripts\python.exe -m pip install -r requirements.txt --no-deps
```

Note: TensorFlow 2.11+ does not provide native Windows GPU support.
If GPU is still not detected, install CUDA 11.2 + cuDNN 8.1 and ensure their `bin` paths are in `PATH`.

Optional runtime flags:

```powershell
$env:USE_GPU = "1"             # 1/0 to enable/disable GPU
$env:GPU_MEMORY_GROWTH = "1"   # avoid pre-allocating all VRAM
$env:MIXED_PRECISION = "1"     # enable mixed_float16 policy
```

### Interpreter Standardization (Important)

- Use only `.venv311` for this project.
- Workspace settings pin the interpreter to `.venv311/Scripts/python.exe` in `.vscode/settings.json`.
- If VS Code still shows unresolved imports, run **Python: Select Interpreter** and pick `.venv311` once.

## Dataset Preparation

Download and extract UIEB into `data/raw` and `data/reference`:

```powershell
.\.venv311\Scripts\python.exe scripts/download_dataset.py
```

If you already have a local dataset, place paired images in:

- `data/raw`
- `data/reference`

Then validate:

```powershell
.\.venv311\Scripts\python.exe scripts/validate_dataset.py --strict-names
```

## Training

Run default training from `config.yaml`:

```powershell
.\.venv311\Scripts\python.exe train_unet.py
```

Run a quick smoke test (1 epoch):

```powershell
.\.venv311\Scripts\python.exe -c "from train_unet import main; main({'epochs': 1, 'batch_size': 2})"
```

## Web Interface (Streamlit)

Install dependencies (if not already installed):

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
.\.venv311\Scripts\python.exe -m streamlit run streamlit_app.py
```

What the app supports:

- Select trained checkpoint from `models/checkpoints`
- Upload image (`jpg`, `jpeg`, `png`, `bmp`)
- Run enhancement inference
- Compare original vs enhanced output
- Download enhanced image as PNG
- View run metadata from `results/model_registry.json`
- Compare two experiment runs with metric deltas and validation curves

## Real-Time Video Processing

Use `video_processor.py` for webcam, video file, RTSP stream, and batch folder enhancement.

Install dependencies (if not already installed):

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

### Webcam mode

```powershell
.\.venv311\Scripts\python.exe video_processor.py --mode webcam
```

Threaded webcam mode (can improve responsiveness on some systems):

```powershell
.\.venv311\Scripts\python.exe video_processor.py --mode webcam --threaded
```

### Video file mode

```powershell
.\.venv311\Scripts\python.exe video_processor.py --mode video --input input.mp4 --output results/processed_videos/input_enhanced.mp4
```

Disable preview window:

```powershell
.\.venv311\Scripts\python.exe video_processor.py --mode video --input input.mp4 --no-preview
```

### RTSP mode

```powershell
.\.venv311\Scripts\python.exe video_processor.py --mode rtsp --input "rtsp://username:password@host:554/stream"
```

Record RTSP output:

```powershell
.\.venv311\Scripts\python.exe video_processor.py --mode rtsp --input "rtsp://..." --output results/processed_videos/rtsp_recording.mp4
```

### Batch mode

```powershell
.\.venv311\Scripts\python.exe video_processor.py --mode batch --input-folder videos --output-folder results/processed_videos
```

### Model selection and performance knobs

- By default, the script auto-selects a checkpoint from `results/model_registry.json` or `models/checkpoints`.
- To use a specific checkpoint, pass `--model models/checkpoints/your_model.keras`.
- Use `--target-size 128` for higher FPS, or `--target-size 256` for balanced quality/speed.
- Use `--no-gpu` to force CPU mode.

### Quick smoke test

```powershell
.\.venv311\Scripts\python.exe quick_video_test.py
```

This generates a synthetic test clip and writes `test_enhanced.mp4`.

## Configuration

Default runtime configuration lives in `config.yaml` and is loaded by `utils/config_loader.py`.

### Augmentation Profiles

You can now control augmentation strength from `config.yaml`:

```yaml
augmentation:
	enabled: true
	profile: standard  # one of: none, light, standard, strong
	flip_prob: 0.5
	vertical_flip_prob: 0.5
	rotate_prob: 1.0
	brightness_prob: 0.5
	brightness_delta: 0.1
	contrast_prob: 0.5
	contrast_lower: 0.8
	contrast_upper: 1.2
```

For A/B tests, keep `profile` fixed and change only one knob at a time (for example `brightness_delta`).

To override dataset location without editing files:

```powershell
$env:DATA_PATH = "D:\datasets\uieb"
.\.venv311\Scripts\python.exe train_unet.py
```