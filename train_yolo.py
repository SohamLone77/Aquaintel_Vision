#!/usr/bin/env python3
"""Train and validate YOLO for underwater threat detection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

try:
    from experiment_tracker import ExperimentTracker
except Exception:
    ExperimentTracker = None


def detect_placeholder_labels(dataset_yaml: Path) -> tuple[bool, dict]:
    """Detect synthetic placeholder labels (single class + near-identical bbox geometry)."""
    with dataset_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", dataset_yaml.parent))
    if not base.is_absolute():
        base = (dataset_yaml.parent / base).resolve()

    class_counter = Counter()
    widths = []
    heights = []
    x_centers = []
    y_centers = []
    total_instances = 0

    for split in ("train", "val"):
        split_rel = cfg.get(split)
        if not split_rel:
            continue

        labels_dir = base / str(split_rel).replace("images", "labels")
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_counter[int(float(parts[0]))] += 1
                x_centers.append(float(parts[1]))
                y_centers.append(float(parts[2]))
                widths.append(float(parts[3]))
                heights.append(float(parts[4]))
                total_instances += 1

    if total_instances == 0:
        return False, {
            "total_instances": 0,
            "class_dist": {},
            "reason": "no_labels_found",
        }

    def _is_near_constant(values, eps=1e-6):
        return (max(values) - min(values)) <= eps if values else False

    placeholder_like = (
        len(class_counter) == 1
        and _is_near_constant(widths)
        and _is_near_constant(heights)
        and _is_near_constant(x_centers)
        and _is_near_constant(y_centers)
    )

    diagnostics = {
        "total_instances": total_instances,
        "class_dist": dict(class_counter),
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "x_center_min": min(x_centers),
        "x_center_max": max(x_centers),
        "y_center_min": min(y_centers),
        "y_center_max": max(y_centers),
    }
    return placeholder_like, diagnostics


class UnderwaterYOLOTrainer:
    """Fine-tune YOLO on an underwater threat dataset."""

    def __init__(
        self,
        dataset_yaml: str = "underwater_dataset/dataset.yaml",
        model_name: str = "yolov8n.pt",
        run_name: str = "underwater_threat_detector",
        project_dir: str = "runs",
        allow_placeholder_labels: bool = False,
    ):
        self.dataset_yaml = Path(dataset_yaml)
        self.model_name = model_name
        self.run_name = run_name
        self.project_dir = project_dir
        self.allow_placeholder_labels = bool(allow_placeholder_labels)
        self.device = 0 if torch.cuda.is_available() else "cpu"

        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.dataset_yaml}")

        placeholder_like, diagnostics = detect_placeholder_labels(self.dataset_yaml)
        if placeholder_like and not self.allow_placeholder_labels:
            raise ValueError(
                "Detected placeholder YOLO labels (single class and near-identical centered boxes). "
                "Replace with real annotations or pass allow_placeholder_labels=True / --allow-placeholder-labels. "
                f"Diagnostics: {diagnostics}"
            )

        with self.dataset_yaml.open("r", encoding="utf-8") as f:
            self.dataset_info = yaml.safe_load(f)

        print("=" * 60)
        print("UNDERWATER YOLO TRAINER")
        print("=" * 60)
        print(f"Dataset: {self.dataset_yaml}")
        print(f"Base model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.dataset_info.get('nc')} -> {self.dataset_info.get('names')}")

    def train(self, epochs: int = 100, imgsz: int = 640, batch: int = 16, patience: int = 20):
        model = YOLO(self.model_name)

        args = {
            "data": str(self.dataset_yaml),
            "epochs": int(epochs),
            "imgsz": int(imgsz),
            "batch": int(batch),
            "patience": int(patience),
            "device": self.device,
            "workers": 4,
            "save": True,
            "save_period": 10,
            "val": True,
            "verbose": True,
            "name": self.run_name,
            "project": self.project_dir,
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "plots": True,
        }

        print("Starting YOLO training...")
        results = model.train(**args)
        print("Training complete.")

        self.save_training_results(results)
        return model, results

    def _find_latest_run_dir(self, results=None) -> Path | None:
        if results is not None:
            try:
                run_dir = Path(results.save_dir)
                if run_dir.exists():
                    return run_dir
            except Exception:
                pass

        candidates = sorted(
            Path(self.project_dir).glob(f"**/{self.run_name}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None

    def _find_best_model_path(self, results=None) -> Path | None:
        run_dir = self._find_latest_run_dir(results=results)
        if run_dir is None:
            return None
        best = run_dir / "weights" / "best.pt"
        return best if best.exists() else None

    def _find_results_csv(self, results=None) -> Path | None:
        run_dir = self._find_latest_run_dir(results=results)
        if run_dir is None:
            return None
        csv_path = run_dir / "results.csv"
        return csv_path if csv_path.exists() else None

    def save_training_results(self, results) -> None:
        out_dir = Path("results/yolo_training")
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self._find_results_csv(results=results)
        if csv_path is not None and csv_path.exists():
            df = pd.read_csv(csv_path)
            df.to_csv(out_dir / "training_metrics.csv", index=False)
            self.plot_training_curves(df, out_dir / "training_curves.png")
            print(f"Saved metrics: {out_dir / 'training_metrics.csv'}")
            print(f"Saved curves:  {out_dir / 'training_curves.png'}")

        self.write_manifest(results=results)

    def write_manifest(self, results=None) -> None:
        run_dir = self._find_latest_run_dir(results=results)
        if run_dir is None:
            return

        best = run_dir / "weights" / "best.pt"
        manifest = {
            "run_dir": str(run_dir).replace("\\", "/"),
            "dataset_yaml": str(self.dataset_yaml).replace("\\", "/"),
            "best_model": str(best.resolve()).replace("\\", "/") if best.exists() else None,
            "classes": self.dataset_info.get("names", []),
        }
        out_path = run_dir / "manifest.json"
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def plot_training_curves(self, df: pd.DataFrame, save_path: Path) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if {"epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss"}.issubset(df.columns):
            axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Box Loss")
            axes[0, 0].plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
            axes[0, 0].plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
            axes[0, 0].set_title("Training Losses")
            axes[0, 0].legend()

        if {"epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"}.issubset(df.columns):
            axes[0, 1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
            axes[0, 1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
            axes[0, 1].set_title("mAP")
            axes[0, 1].legend()

        if {"epoch", "metrics/precision(B)", "metrics/recall(B)"}.issubset(df.columns):
            axes[1, 0].plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
            axes[1, 0].plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
            axes[1, 0].set_title("Precision / Recall")
            axes[1, 0].legend()

        if {"epoch", "lr/pg0"}.issubset(df.columns):
            axes[1, 1].plot(df["epoch"], df["lr/pg0"], label="Learning Rate")
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].legend()

        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Epoch")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

    def validate_model(self):
        best_model_path = self._find_best_model_path()
        if best_model_path is None:
            print(f"Best model not found: {best_model_path}")
            return None

        model = YOLO(str(best_model_path))
        results = model.val(data=str(self.dataset_yaml))

        print("Validation Results:")
        print(f"mAP50:     {results.box.map50:.4f}")
        print(f"mAP50-95:  {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall:    {results.box.mr:.4f}")
        return results

    def export_model(self, fmt: str = "onnx"):
        best_model_path = self._find_best_model_path()
        if best_model_path is None:
            print(f"Cannot export, model not found: {best_model_path}")
            return

        model = YOLO(str(best_model_path))
        model.export(format=fmt, imgsz=640)
        print(f"Exported model to format: {fmt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO for underwater threat detection")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch count")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--run-name", type=str, default="underwater_threat_detector", help="Run name under project directory")
    parser.add_argument("--project", type=str, default="runs", help="Parent output directory for YOLO runs")
    parser.add_argument("--data", type=str, default="underwater_dataset/dataset.yaml", help="Dataset yaml path")
    parser.add_argument(
        "--allow-placeholder-labels",
        action="store_true",
        help="Allow training on synthetic/placeholder labels (not recommended).",
    )
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--export", type=str, default=None, help="Export format: onnx/tflite/torchscript")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trainer = UnderwaterYOLOTrainer(
        dataset_yaml=args.data,
        model_name=args.model,
        run_name=args.run_name,
        project_dir=args.project,
        allow_placeholder_labels=args.allow_placeholder_labels,
    )

    if args.validate:
        trainer.validate_model()
        return

    if args.export:
        trainer.export_model(fmt=args.export)
        return

    trainer.train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience)
    val_results = trainer.validate_model()

    if ExperimentTracker is not None and val_results is not None:
        try:
            best_model = trainer._find_best_model_path()
            tracker = ExperimentTracker()
            tracker.register_yolo_experiment(
                run_id=(best_model.parent.parent.name if best_model is not None else "underwater_threat_detector"),
                metrics={
                    "mAP50": float(val_results.box.map50),
                    "precision": float(val_results.box.mp),
                    "recall": float(val_results.box.mr),
                },
                model_path=str(best_model) if best_model is not None else "",
                is_promoted=False,
            )
            print("Experiment tracker updated for YOLO run")
        except Exception as exc:
            print(f"Experiment tracker update skipped: {exc}")

    trainer.export_model(fmt="onnx")


if __name__ == "__main__":
    main()
