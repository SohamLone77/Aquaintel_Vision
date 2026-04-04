# AI

Underwater image enhancement training workspace based on a U-Net pipeline.

## Current Status

- Step 1 completed: centralized runtime config via `config.yaml`.
- Step 2 completed: dataset validation enforced before training starts.
- Step 3 completed: deterministic dataset download and extract script.
- Environment milestone completed: Python 3.11 venv with dependencies installed.

Progress tracker: see `IMPLEMENTATION_TODO.md`.

## Environment Setup

Create and install dependencies in a Python 3.11 virtual environment:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

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