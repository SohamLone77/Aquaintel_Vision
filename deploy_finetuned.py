#!/usr/bin/env python3
"""Deploy a fine-tuned YOLO model with per-class thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


class UnderwaterThreatDetector:
    def __init__(self, model_path: str = "runs/finetune/underwater_finetuned/weights/best.pt"):
        self.model = YOLO(model_path)

        # Lower thresholds on critical low-recall classes
        self.class_thresholds = {
            "diver": 0.25,
            "suspicious_object": 0.20,
            "swimmer": 0.20,
            "underwater_vehicle": 0.20,
            "obstacle": 0.30,
            "boat": 0.30,
            "ship": 0.30,
            "submarine": 0.25,
            "pipeline": 0.35,
            "marine_life": 0.35,
        }

    def detect(self, image_bgr, conf: float = 0.20):
        results = self.model(image_bgr, conf=conf, verbose=False)[0]

        detections = []
        for box in results.boxes:
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            if score < self.class_thresholds.get(class_name, 0.25):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {
                    "class": class_name,
                    "confidence": score,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return detections

    def annotate(self, image_bgr, detections: list[dict]):
        out = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(out, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deployed fine-tuned underwater detector")
    parser.add_argument("--model", default="runs/finetune/underwater_finetuned/weights/best.pt", help="Fine-tuned model path")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Annotated output image path")
    parser.add_argument("--conf", type=float, default=0.20, help="Base confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detector = UnderwaterThreatDetector(model_path=args.model)

    image_path = Path(args.input)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {image_path}")

    detections = detector.detect(image_bgr=image, conf=float(args.conf))
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f"  - {det['class']}: {det['confidence']:.3f} @ {det['bbox']}")

    annotated = detector.annotate(image, detections)
    output_path = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_finetuned_detected{image_path.suffix}")
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved annotated image: {output_path}")


if __name__ == "__main__":
    main()
