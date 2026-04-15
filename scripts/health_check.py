#!/usr/bin/env python3
"""Project health-check to catch regressions before long runs."""

import glob
import py_compile
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_dataset import validate_dataset


KEY_FILES = [
    PROJECT_ROOT / "train_unet.py",
    PROJECT_ROOT / "train_complete.py",
    PROJECT_ROOT / "train_sharp.py",
    PROJECT_ROOT / "resume_sharp.py",
    PROJECT_ROOT / "sharpen_output.py",
    PROJECT_ROOT / "compare_results.py",
    PROJECT_ROOT / "streamlit_app.py",
]


def run_py_compile_checks() -> None:
    print("[1/3] Running py_compile checks...")
    for file_path in KEY_FILES:
        py_compile.compile(str(file_path), doraise=True)
        print(f"  OK: {file_path.name}")


def run_dataset_validation() -> None:
    print("[2/3] Running dataset validation...")
    is_valid, details = validate_dataset(data_path="data")
    if not is_valid:
        issues = "\n  - ".join(details.get("issues", []))
        raise RuntimeError(f"Dataset validation failed:\n  - {issues}")

    paired_count = int(details.get("paired_count", 0))
    if paired_count <= 0:
        raise RuntimeError("Dataset validation returned zero matched pairs")

    print(f"  OK: {paired_count} matched pairs detected")


def run_inference_smoke() -> None:
    print("[3/3] Running model load/inference smoke...")
    candidates = sorted(glob.glob(str(PROJECT_ROOT / "models" / "checkpoints" / "*.h5")))
    if not candidates:
        raise FileNotFoundError("No .h5 checkpoints found in models/checkpoints")

    latest_model_path = max(candidates, key=lambda p: Path(p).stat().st_mtime)
    model = tf.keras.models.load_model(latest_model_path, compile=False)
    input_size = int(model.input_shape[1])

    x = np.random.rand(1, input_size, input_size, 3).astype(np.float32)
    y = model.predict(x, verbose=0)

    if y is None or y.shape[0] != 1:
        raise RuntimeError("Inference smoke failed: invalid output tensor")

    print(f"  OK: loaded {Path(latest_model_path).name}")
    print(f"  OK: input {x.shape} -> output {y.shape}")


def main() -> None:
    print("=" * 60)
    print("PROJECT HEALTH CHECK")
    print("=" * 60)

    run_py_compile_checks()
    run_dataset_validation()
    run_inference_smoke()

    print("\nHealth check passed.")


if __name__ == "__main__":
    main()
