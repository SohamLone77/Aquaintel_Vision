"""Validate YOLO annotations after manual labeling."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2


CLASS_NAMES = {
    0: "diver",
    1: "suspicious_object",
    2: "boat",
    3: "ship",
    4: "submarine",
    5: "pipeline",
    6: "obstacle",
    7: "marine_life",
    8: "swimmer",
    9: "underwater_vehicle",
}


def _list_images(image_dir: Path) -> list[Path]:
    images: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        images.extend(image_dir.glob(ext))
    return sorted(set(images))


def validate_annotations(split: str = "train") -> tuple[int, int]:
    image_dir = Path(f"underwater_dataset/images/{split}")
    label_dir = Path(f"underwater_dataset/labels/{split}")

    images = _list_images(image_dir)
    labels = [p for p in sorted(label_dir.glob("*.txt")) if p.name.lower() != "classes.txt"]

    print("\n" + "=" * 60)
    print(f"VALIDATING {split.upper()} SPLIT")
    print("=" * 60)
    print(f"Images: {len(images)}")
    print(f"Label files: {len(labels)}")

    images_without_labels: list[str] = []
    for img in images:
        label_path = label_dir / f"{img.stem}.txt"
        if not label_path.exists():
            images_without_labels.append(img.name)

    if images_without_labels:
        print(f"\nWARNING: Missing labels for {len(images_without_labels)} images:")
        for name in images_without_labels[:20]:
            print(f"  - {name}")

    class_counts: Counter[int] = Counter()
    total_objects = 0
    valid_labels = 0
    invalid_labels = 0

    for label_path in labels:
        for line_num, line in enumerate(label_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"WARNING: Invalid format in {label_path.name}:{line_num}")
                invalid_labels += 1
                continue

            try:
                class_id = int(float(parts[0]))
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                if class_id not in CLASS_NAMES:
                    print(f"WARNING: Unknown class id in {label_path.name}:{line_num} -> {class_id}")
                    invalid_labels += 1
                    continue

                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    print(f"WARNING: Out-of-range bbox values in {label_path.name}:{line_num}")
                    invalid_labels += 1
                    continue

                class_counts[class_id] += 1
                total_objects += 1
                valid_labels += 1
            except ValueError as exc:
                print(f"WARNING: Parse error in {label_path.name}:{line_num}: {exc}")
                invalid_labels += 1

    print("\nStatistics:")
    print(f"  Total objects annotated: {total_objects}")
    print(f"  Valid labels: {valid_labels}")
    print(f"  Invalid labels: {invalid_labels}")

    print("\nClass Distribution:")
    for class_id, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {CLASS_NAMES[class_id]}: {count} objects")

    return valid_labels, invalid_labels


def verify_single_label(split: str, image_name: str) -> bool:
    image_path = Path(f"underwater_dataset/images/{split}/{image_name}")
    label_path = Path(f"underwater_dataset/labels/{split}/{Path(image_name).stem}.txt")

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return False

    if not label_path.exists():
        print(f"ERROR: Label file not found: {label_path}")
        return False

    h, w = img.shape[:2]

    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    print(f"\nImage: {image_path.name}")
    print(f"Labels: {len(lines)} objects")

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        width = float(parts[3]) * w
        height = float(parts[4]) * h

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        print(f"  {CLASS_NAMES.get(class_id, f'class_{class_id}')}: box=({x1},{y1},{x2},{y2})")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Annotation Check", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO annotations")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--verify", type=str, default=None, help="Specific image file name to verify")
    args = parser.parse_args()

    if args.verify:
        verify_single_label(args.split, args.verify)
    else:
        validate_annotations(args.split)


if __name__ == "__main__":
    main()
