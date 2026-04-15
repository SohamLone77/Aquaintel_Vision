#!/usr/bin/env python3
"""Compare baseline and fine-tuned YOLO models on a sample set."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def gather_images(test_dir: Path, limit: int) -> list[Path]:
    images: list[Path] = []
    for pattern in IMAGE_EXTS:
        images.extend(test_dir.glob(pattern))
    images = sorted(images)
    return images[:limit] if limit > 0 else images


def compare_models(
    baseline_path: str,
    finetuned_path: str,
    test_images: list[Path],
    conf: float,
) -> pd.DataFrame:
    baseline = YOLO(baseline_path)
    finetuned = YOLO(finetuned_path)

    rows: list[dict] = []

    for img_path in test_images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        base_result = baseline(image, conf=conf, verbose=False)[0]
        ft_result = finetuned(image, conf=conf, verbose=False)[0]

        base_confs = [float(box.conf[0]) for box in base_result.boxes]
        ft_confs = [float(box.conf[0]) for box in ft_result.boxes]

        rows.append(
            {
                "image": img_path.name,
                "baseline_detections": len(base_result.boxes),
                "finetuned_detections": len(ft_result.boxes),
                "baseline_mean_conf": sum(base_confs) / len(base_confs) if base_confs else 0.0,
                "finetuned_mean_conf": sum(ft_confs) / len(ft_confs) if ft_confs else 0.0,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned YOLO models")
    parser.add_argument("--baseline", required=True, help="Baseline model path")
    parser.add_argument("--finetuned", required=True, help="Fine-tuned model path")
    parser.add_argument("--test-dir", default="data/sample/raw", help="Directory of test images")
    parser.add_argument("--limit", type=int, default=20, help="Max test images (0 = all)")
    parser.add_argument("--conf", type=float, default=0.20, help="Inference confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    test_images = gather_images(test_dir, args.limit)
    if not test_images:
        print(f"No test images found in {test_dir}")
        return

    df = compare_models(
        baseline_path=args.baseline,
        finetuned_path=args.finetuned,
        test_images=test_images,
        conf=float(args.conf),
    )

    if df.empty:
        print("No comparisons produced")
        return

    out_dir = Path("results/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"model_comparison_{ts}.csv"
    df.to_csv(out_csv, index=False)

    baseline_total = int(df["baseline_detections"].sum())
    finetuned_total = int(df["finetuned_detections"].sum())

    print("Model Comparison Summary")
    print(f"Images compared: {len(df)}")
    print(f"Baseline total detections:  {baseline_total}")
    print(f"Fine-tuned total detections: {finetuned_total}")
    print(f"Saved detailed report: {out_csv}")


if __name__ == "__main__":
    main()
