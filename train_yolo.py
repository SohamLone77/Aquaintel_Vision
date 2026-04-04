#!/usr/bin/env python3
"""Train and validate YOLO for underwater threat detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


class UnderwaterYOLOTrainer:
    """Fine-tune YOLO on an underwater threat dataset."""

    def __init__(self, dataset_yaml: str = "underwater_dataset/dataset.yaml", model_name: str = "yolov8n.pt"):
        self.dataset_yaml = Path(dataset_yaml)
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else "cpu"

        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.dataset_yaml}")

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
            "name": "underwater_threat_detector",
            "project": "runs",
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

        candidates = sorted(Path("runs").glob("**/underwater_threat_detector"), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in candidates:
            if c.is_dir():
                return c
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
    parser.add_argument("--data", type=str, default="underwater_dataset/dataset.yaml", help="Dataset yaml path")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--export", type=str, default=None, help="Export format: onnx/tflite/torchscript")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trainer = UnderwaterYOLOTrainer(dataset_yaml=args.data, model_name=args.model)

    if args.validate:
        trainer.validate_model()
        return

    if args.export:
        trainer.export_model(fmt=args.export)
        return

    trainer.train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience)
    trainer.validate_model()
    trainer.export_model(fmt="onnx")


if __name__ == "__main__":
    main()
