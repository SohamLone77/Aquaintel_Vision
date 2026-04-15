#!/usr/bin/env python3
"""Quick batch evaluation helper for the production detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from production_detector import DEFAULT_YOLO_PATH, UnderwaterThreatDetector


IMG_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")


def collect_images(folder: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in IMG_PATTERNS:
        images.extend(sorted(folder.glob(pattern)))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate production detector")
    parser.add_argument("--input-dir", default="underwater_dataset/images/val", help="Image directory")
    parser.add_argument("--output", default="results/batch_evaluation.json", help="Output JSON")
    parser.add_argument("--model", default=DEFAULT_YOLO_PATH, help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--enhance", action="store_true", help="Enable enhancement")
    parser.add_argument("--limit", type=int, default=40, help="Max images to evaluate")
    args = parser.parse_args()

    detector = UnderwaterThreatDetector(
        yolo_model_path=args.model,
        confidence_threshold=args.conf,
        enable_enhancement=args.enhance,
    )

    image_dir = Path(args.input_dir)
    images = collect_images(image_dir)[: max(1, int(args.limit))]

    results = []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        detections, _, elapsed_ms = detector.detect(image)
        results.append(
            {
                "image": img_path.name,
                "detections": len(detections),
                "classes": [d["class_name"] for d in detections],
                "inference_ms": round(elapsed_ms, 3),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    total_detections = sum(r["detections"] for r in results)
    print(f"Evaluated {len(results)} images")
    print(f"Total detections: {total_detections}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
