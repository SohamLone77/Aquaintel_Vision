"""Visualize annotation quality by rendering label overlays on image grids."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


CLASS_COLORS = {
    0: (255, 0, 0),
    1: (255, 128, 0),
    2: (255, 255, 0),
    3: (0, 255, 0),
    4: (0, 0, 255),
    5: (128, 0, 255),
    6: (0, 255, 255),
    7: (255, 0, 255),
    8: (128, 64, 0),
    9: (255, 128, 192),
}


def _list_images(image_dir: Path) -> list[Path]:
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        images.extend(image_dir.glob(ext))
    return sorted(set(images))


def visualize_all_annotations(split: str = "train", max_images: int = 20) -> None:
    image_dir = Path(f"underwater_dataset/images/{split}")
    label_dir = Path(f"underwater_dataset/labels/{split}")

    images = _list_images(image_dir)[:max_images]
    if not images:
        print(f"No images found for split: {split}")
        return

    cols = 4
    rows = (len(images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, img_path in enumerate(images):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            axes[idx].axis("off")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_w = float(parts[3]) * w
                box_h = float(parts[4]) * h

                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)

                color = CLASS_COLORS.get(class_id, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        axes[idx].imshow(img)
        axes[idx].set_title(f"{img_path.name}\n{label_path.name if label_path.exists() else 'NO LABEL'}")
        axes[idx].axis("off")

    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    out_path = Path(f"annotation_quality_check_{split}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    visualize_all_annotations(split="train", max_images=12)
    visualize_all_annotations(split="val", max_images=8)


if __name__ == "__main__":
    main()
