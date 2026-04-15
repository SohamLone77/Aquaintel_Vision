"""Track annotation progress for YOLO train/val splits."""

from __future__ import annotations

import json
from pathlib import Path


def _list_images(image_dir: Path) -> list[Path]:
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        images.extend(image_dir.glob(ext))
    return sorted(set(images))


def track_progress() -> dict:
    splits = ["train", "val"]
    progress = {}

    for split in splits:
        image_dir = Path(f"underwater_dataset/images/{split}")
        label_dir = Path(f"underwater_dataset/labels/{split}")

        images = _list_images(image_dir)
        labels = sorted(label_dir.glob("*.txt"))

        # Placeholder pattern currently used in this repo's synthetic labels.
        dummy_pattern = "0 0.5 0.5 0.3 0.4"

        real_labels = 0
        for label_path in labels:
            content = label_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            if dummy_pattern not in content:
                real_labels += 1

        total_images = len(images)
        progress[split] = {
            "total_images": total_images,
            "total_labels": len(labels),
            "real_annotations": real_labels,
            "dummy_annotations": max(0, len(labels) - real_labels),
            "progress_percent": (real_labels / total_images * 100.0) if total_images else 0.0,
        }

    out_path = Path("annotation_progress.json")
    out_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("ANNOTATION PROGRESS")
    print("=" * 50)
    for split, data in progress.items():
        print(f"\n{split.upper()}:")
        print(f"   Images: {data['total_images']}")
        print(f"   Annotated: {data['real_annotations']}/{data['total_images']}")
        print(f"   Progress: {data['progress_percent']:.1f}%")
        if data["dummy_annotations"] > 0:
            print(f"   WARNING: Dummy labels remaining: {data['dummy_annotations']}")

    print(f"\nSaved: {out_path}")
    return progress


if __name__ == "__main__":
    track_progress()
