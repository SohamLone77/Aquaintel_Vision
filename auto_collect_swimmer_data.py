#!/usr/bin/env python3
"""Collect swimmer candidate images using a COCO pre-trained detector."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-collect swimmer candidate frames")
    parser.add_argument("--model", default="yolov8n.pt", help="COCO pre-trained model")
    parser.add_argument("--unlabeled-dir", default="underwater_dataset/images/unlabeled", help="Unlabeled image directory")
    parser.add_argument("--output-dir", default="swimmer_candidates", help="Output directory for candidate images")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--save-boxes", action="store_true", help="Save candidate person boxes as YOLO txt sidecars")
    return parser.parse_args()


def gather_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in IMAGE_EXTS:
        images.extend(root.glob(pattern))
    return sorted(images)


def main() -> None:
    args = parse_args()

    unlabeled_dir = Path(args.unlabeled_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not unlabeled_dir.exists():
        raise FileNotFoundError(f"Unlabeled directory not found: {unlabeled_dir}")

    model = YOLO(args.model)
    images = gather_images(unlabeled_dir)

    if not images:
        print(f"No images found in {unlabeled_dir}")
        return

    total_candidates = 0
    for img_path in images:
        result = model(str(img_path), conf=float(args.conf), verbose=False)[0]

        # COCO class 0 = person (proxy candidate for swimmer)
        persons = [box for box in result.boxes if int(box.cls[0]) == 0]
        if not persons:
            continue

        dst_img = output_dir / img_path.name
        shutil.copy2(img_path, dst_img)

        if args.save_boxes:
            # Save in YOLO normalized format, class 0 (candidate swimmer placeholder)
            h, w = result.orig_shape
            lines = []
            for box in persons:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_c = ((x1 + x2) / 2.0) / w
                y_c = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

            (output_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        total_candidates += 1
        print(f"Candidate swimmer frame: {img_path.name} ({len(persons)} person detections)")

    print("Collection complete")
    print(f"Input images scanned: {len(images)}")
    print(f"Candidate images found: {total_candidates}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
