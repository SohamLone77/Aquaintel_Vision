#!/usr/bin/env python3
"""Prepare a YOLO-style underwater dataset with train/val/test splits."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
import yaml

try:
    from dataset_integrity import DatasetIntegrityChecker
except Exception:
    DatasetIntegrityChecker = None


class UnderwaterDatasetPreparer:
    """Prepare underwater dataset for YOLO fine-tuning."""

    def __init__(self, source_dir: str = "data/sample", output_dir: str = "underwater_dataset", seed: int = 42):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.raw_dir = self.source_dir / "raw"
        self.ref_dir = self.source_dir / "reference"
        self.seed = int(seed)

        self.classes = {
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

    def create_directories(self) -> None:
        dirs = [
            self.output_dir,
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _image_paths(self) -> list[Path]:
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        images: list[Path] = []
        for ext in exts:
            images.extend(self.raw_dir.glob(ext))
            images.extend(self.raw_dir.glob(ext.upper()))
        return sorted(set(images))

    def _write_dummy_label(self, label_path: Path) -> None:
        # YOLO format: class_id x_center y_center width height (normalized)
        label_path.write_text("0 0.5 0.5 0.3 0.4\n", encoding="utf-8")

    def _copy_and_label(self, image_paths: list[Path], split: str, create_labels: bool) -> None:
        for img_path in image_paths:
            dst_image = self.output_dir / "images" / split / img_path.name
            shutil.copy2(img_path, dst_image)

            dst_label = self.output_dir / "labels" / split / f"{img_path.stem}.txt"
            if create_labels:
                self._write_dummy_label(dst_label)
            else:
                dst_label.write_text("", encoding="utf-8")

    def create_sample_annotations(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
        images = self._image_paths()
        if not images:
            raise FileNotFoundError(
                f"No images found in {self.raw_dir}. Add files under data/sample/raw/ first."
            )

        if train_ratio <= 0 or val_ratio <= 0 or (train_ratio + val_ratio) >= 1:
            raise ValueError("Invalid split ratios. Use train>0, val>0 and train+val<1.")

        random.seed(self.seed)

        train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=self.seed)
        # If val_ratio is 0.15 and test is 0.15, val fraction over temp is 0.5
        val_fraction_of_temp = val_ratio / (1 - train_ratio)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(1 - val_fraction_of_temp), random_state=self.seed)

        self._copy_and_label(train_imgs, "train", create_labels=True)
        self._copy_and_label(val_imgs, "val", create_labels=True)
        self._copy_and_label(test_imgs, "test", create_labels=False)

        return {
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
            "total": len(images),
        }

    def create_dataset_yaml(self) -> Path:
        cfg = {
            "path": str(self.output_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": [self.classes[i] for i in sorted(self.classes.keys())],
        }
        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return yaml_path

    def generate_annotation_guide(self) -> Path:
        guide = (
            "UNDERWATER THREAT DETECTION - ANNOTATION GUIDE\n"
            "=============================================\n\n"
            "Use LabelImg/CVAT/Roboflow to replace demo labels.\n"
            "YOLO label format: class_id x_center y_center width height\n"
            "Values are normalized to [0, 1].\n\n"
            "Classes:\n"
            + "\n".join([f"{k}: {v}" for k, v in self.classes.items()])
            + "\n"
        )
        path = self.output_dir / "ANNOTATION_GUIDE.txt"
        path.write_text(guide, encoding="utf-8")
        return path

    def write_summary(self, split_stats: dict, yaml_path: Path) -> Path:
        summary = {
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "split": split_stats,
            "dataset_yaml": str(yaml_path),
            "note": "train/val labels are demo placeholders and must be replaced by real annotations.",
        }
        out = self.output_dir / "prep_summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out

    def prepare(self, run_integrity_check: bool = False) -> Path:
        print("=" * 60)
        print("UNDERWATER DATASET PREPARER")
        print("=" * 60)
        self.create_directories()
        split_stats = self.create_sample_annotations()
        yaml_path = self.create_dataset_yaml()
        guide_path = self.generate_annotation_guide()
        summary_path = self.write_summary(split_stats, yaml_path)

        print(f"Dataset ready: {self.output_dir}")
        print(f"Split -> train: {split_stats['train']}, val: {split_stats['val']}, test: {split_stats['test']}")
        print(f"Config: {yaml_path}")
        print(f"Guide: {guide_path}")
        print(f"Summary: {summary_path}")
        print("Next: replace dummy labels and run train_yolo.py")

        if run_integrity_check:
            if DatasetIntegrityChecker is None:
                print("Integrity checker unavailable; skipping integrity validation.")
            else:
                checker = DatasetIntegrityChecker(dataset_path=str(self.output_dir), backup=False)
                train_ok = checker.validate_all(split="train")
                val_ok = checker.validate_all(split="val")
                print(f"Integrity check -> train: {train_ok}, val: {val_ok}")
        return yaml_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare underwater dataset for YOLO training")
    parser.add_argument("--source-dir", type=str, default="data/sample", help="Input source directory")
    parser.add_argument("--output-dir", type=str, default="underwater_dataset", help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--integrity-check", action="store_true", help="Run dataset integrity validation after preparation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preparer = UnderwaterDatasetPreparer(source_dir=args.source_dir, output_dir=args.output_dir, seed=args.seed)
    preparer.prepare(run_integrity_check=args.integrity_check)


if __name__ == "__main__":
    main()
