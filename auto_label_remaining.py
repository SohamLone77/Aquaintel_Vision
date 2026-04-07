"""Auto-label remaining YOLO images using a trained detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def list_images(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        images.extend(image_dir.glob(ext))
    return sorted(set(images))


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label unlabeled images with a YOLO model")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO .pt model")
    parser.add_argument("--image-dir", type=str, default="underwater_dataset/images/train")
    parser.add_argument("--label-dir", type=str, default="underwater_dataset/labels/train")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    parser.add_argument("--overwrite-empty-only", action="store_true", help="Only generate labels for missing or empty label files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be labeled without writing files")
    args = parser.parse_args()

    model_path = Path(args.model)
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    label_dir.mkdir(parents=True, exist_ok=True)

    all_images = list_images(image_dir)
    if not all_images:
        print(f"No images found in: {image_dir}")
        return

    unlabeled: list[Path] = []
    for img_path in all_images:
        lbl = label_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            unlabeled.append(img_path)
            continue
        if args.overwrite_empty_only and lbl.stat().st_size == 0:
            unlabeled.append(img_path)

    print(f"Total images: {len(all_images)}")
    print(f"Candidates for auto-labeling: {len(unlabeled)}")

    if args.dry_run:
        for p in unlabeled[:20]:
            print(f"DRY_RUN: {p.name}")
        if len(unlabeled) > 20:
            print(f"... and {len(unlabeled) - 20} more")
        return

    model = YOLO(str(model_path))

    written = 0
    empty_written = 0
    total_boxes = 0

    for img_path in unlabeled:
        result = model(str(img_path), conf=args.conf, verbose=False)[0]
        label_path = label_dir / f"{img_path.stem}.txt"

        lines = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
        total_boxes += len(lines)
        if not lines:
            empty_written += 1

        print(f"Labeled {img_path.name}: {len(lines)} boxes")

    print("\nAuto-labeling complete")
    print(f"Files written: {written}")
    print(f"Total boxes: {total_boxes}")
    print(f"Empty label files: {empty_written}")


if __name__ == "__main__":
    main()
