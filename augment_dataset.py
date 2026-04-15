#!/usr/bin/env python3
"""Create augmented YOLO training samples, with focus on rare classes."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
    ALBUMENTATIONS_IMPORT_ERROR = ""
except Exception as exc:
    A = None
    ALBUMENTATIONS_AVAILABLE = False
    ALBUMENTATIONS_IMPORT_ERROR = str(exc)


class UnderwaterAugmenter:
    def __init__(self, image_dir: str, label_dir: str, output_dir: str):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)

        self.out_images = self.output_dir / "images" / "train"
        self.out_labels = self.output_dir / "labels" / "train"
        self.out_images.mkdir(parents=True, exist_ok=True)
        self.out_labels.mkdir(parents=True, exist_ok=True)

        self.transform = None
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = A.Compose(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
                    A.ChannelShuffle(p=0.3),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.5, p=0.4),
                    A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomScale(scale_limit=0.2, p=0.4),
                    A.RandomCrop(height=512, width=512, p=0.3),
                    A.GaussNoise(var_limit=(10.0, 30.0), p=0.4),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.2),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.2,
                ),
            )
            print("Albumentations backend: enabled")
        else:
            print("Albumentations backend unavailable, using OpenCV fallback transforms")
            print(f"Reason: {ALBUMENTATIONS_IMPORT_ERROR}")

    @staticmethod
    def _read_yolo_label(label_path: Path) -> tuple[list[list[float]], list[int]]:
        bboxes: list[list[float]] = []
        class_labels: list[int] = []

        if not label_path.exists():
            return bboxes, class_labels

        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)

        return bboxes, class_labels

    @staticmethod
    def _write_yolo_label(label_path: Path, bboxes: list[list[float]], class_labels: list[int]) -> None:
        lines = []
        for bbox, label in zip(bboxes, class_labels):
            x_c, y_c, w, h = bbox
            lines.append(f"{label} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _fallback_augment(
        self,
        image_rgb: np.ndarray,
        bboxes: list[list[float]],
        class_labels: list[int],
    ) -> tuple[np.ndarray, list[list[float]], list[int]]:
        """Fallback augmentation using OpenCV only.

        This keeps bbox math simple by applying only flip geometry; color/noise/blur
        operations do not change boxes.
        """

        out = image_rgb.copy()
        out_bboxes = [list(b) for b in bboxes]
        out_labels = list(class_labels)

        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-20, 20)
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        if random.random() < 0.5:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] + random.uniform(-20, 20), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] + random.uniform(-10, 10), 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if random.random() < 0.3:
            sigma = random.uniform(5.0, 15.0)
            noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.2:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        if random.random() < 0.3:
            lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

        if random.random() < 0.5:
            out = cv2.flip(out, 1)
            for box in out_bboxes:
                box[0] = 1.0 - box[0]

        if random.random() < 0.3:
            out = cv2.flip(out, 0)
            for box in out_bboxes:
                box[1] = 1.0 - box[1]

        return out, out_bboxes, out_labels

    def augment_image(self, image_path: Path, label_path: Path, num_augmentations: int = 5) -> int:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return 0

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = self._read_yolo_label(label_path)
        if not bboxes:
            return 0

        created = 0
        for i in range(num_augmentations):
            if self.transform is not None:
                augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented["image"]
                aug_bboxes = list(augmented["bboxes"])
                aug_labels = list(augmented["class_labels"])
            else:
                aug_img, aug_bboxes, aug_labels = self._fallback_augment(image, bboxes, class_labels)

            if not aug_bboxes:
                continue

            aug_stem = f"{image_path.stem}_aug{i}"
            aug_image_path = self.out_images / f"{aug_stem}{image_path.suffix}"
            aug_label_path = self.out_labels / f"{aug_stem}.txt"

            aug_image_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_image_path), aug_image_bgr)
            self._write_yolo_label(aug_label_path, aug_bboxes, aug_labels)
            created += 1

        return created

    def run(
        self,
        target_classes: set[int],
        num_augmentations: int,
        include_non_target: bool,
        non_target_augmentations: int,
        copy_originals: bool,
    ) -> None:
        image_paths = sorted(
            [
                p
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                for p in self.image_dir.glob(ext)
            ]
        )

        if not image_paths:
            print(f"No images found in {self.image_dir}")
            return

        total_aug = 0
        total_files = 0

        for image_path in image_paths:
            label_path = self.label_dir / f"{image_path.stem}.txt"
            bboxes, class_labels = self._read_yolo_label(label_path)
            if not bboxes:
                continue

            classes_in_file = set(class_labels)
            has_target = bool(classes_in_file.intersection(target_classes))

            if has_target:
                n_aug = num_augmentations
            elif include_non_target:
                n_aug = non_target_augmentations
            else:
                n_aug = 0

            if copy_originals:
                shutil.copy2(image_path, self.out_images / image_path.name)
                shutil.copy2(label_path, self.out_labels / label_path.name)

            if n_aug > 0:
                created = self.augment_image(image_path, label_path, num_augmentations=n_aug)
                total_aug += created
                total_files += 1
                print(f"Augmented {image_path.name}: +{created}")

        print("Augmentation complete")
        print(f"Source files processed: {total_files}")
        print(f"Augmented samples created: {total_aug}")
        print(f"Output images: {self.out_images}")
        print(f"Output labels: {self.out_labels}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment underwater YOLO training data")
    parser.add_argument("--image-dir", default="underwater_dataset/images/train", help="Source image directory")
    parser.add_argument("--label-dir", default="underwater_dataset/labels/train", help="Source label directory")
    parser.add_argument("--output-dir", default="underwater_dataset_augmented", help="Output dataset directory")
    parser.add_argument(
        "--target-classes",
        nargs="+",
        type=int,
        default=[8, 9],
        help="Class IDs to prioritize (default: swimmer=8, underwater_vehicle=9)",
    )
    parser.add_argument("--num-augmentations", type=int, default=8, help="Augmentations per target image")
    parser.add_argument("--include-non-target", action="store_true", help="Also augment non-target-class images")
    parser.add_argument("--non-target-augmentations", type=int, default=2, help="Augmentations per non-target image")
    parser.add_argument("--no-copy-originals", action="store_true", help="Do not copy originals into output dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    augmenter = UnderwaterAugmenter(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
    )

    augmenter.run(
        target_classes=set(args.target_classes),
        num_augmentations=max(0, int(args.num_augmentations)),
        include_non_target=bool(args.include_non_target),
        non_target_augmentations=max(0, int(args.non_target_augmentations)),
        copy_originals=not args.no_copy_originals,
    )


if __name__ == "__main__":
    main()
