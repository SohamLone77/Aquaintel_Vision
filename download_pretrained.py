#!/usr/bin/env python3
"""Download pre-trained YOLO models for fine-tuning."""

from __future__ import annotations

import argparse

from ultralytics import YOLO

MODEL_MAP = {
    "n": ("yolov8n.pt", "YOLOv8n loaded"),
    "s": ("yolov8s.pt", "YOLOv8s loaded"),
    "m": ("yolov8m.pt", "YOLOv8m loaded"),
    "l": ("yolov8l.pt", "YOLOv8l loaded"),
    "x": ("yolov8x.pt", "YOLOv8x loaded"),
    "11n": ("yolo11n.pt", "YOLOv11n loaded"),
    "11s": ("yolo11s.pt", "YOLOv11s loaded"),
    "11m": ("yolo11m.pt", "YOLOv11m loaded"),
    "11l": ("yolo11l.pt", "YOLOv11l loaded"),
    "11x": ("yolo11x.pt", "YOLOv11x loaded"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download pre-trained YOLO checkpoints (v8/v11)")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_MAP.keys()),
        default=["m", "l", "x"],
        help="Model sizes to download (e.g., n s m l x 11s 11m). Default: m l x",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Downloading requested pre-trained YOLO models...")

    for key in args.models:
        model_id, label = MODEL_MAP[key]
        _ = YOLO(model_id)
        print(f"OK: {label} ({model_id})")

    print("Done.")


if __name__ == "__main__":
    main()
