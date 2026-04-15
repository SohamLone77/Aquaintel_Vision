#!/usr/bin/env python3
"""Fine-tune YOLO for underwater threat detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


SUPPORTED_TRAIN_KEYS = {
    "data",
    "epochs",
    "patience",
    "batch",
    "imgsz",
    "lr0",
    "lrf",
    "optimizer",
    "weight_decay",
    "freeze",
    "mosaic",
    "copy_paste",
    "mixup",
    "box",
    "cls",
    "dfl",
    "warmup_epochs",
    "warmup_momentum",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "auto_augment",
    "device",
    "name",
    "project",
    "exist_ok",
    "verbose",
    "plots",
    "workers",
}


class UnderwaterFineTuner:
    def __init__(self, base_model: str, config_path: str):
        self.base_model = base_model
        self.config_path = Path(config_path)
        self.config = self.load_config(self.config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Initializing fine-tuner")
        print(f"Base model: {self.base_model}")
        print(f"Device: {self.device}")

        if "cls_weights" in self.config:
            print("Note: cls_weights is kept for reference and is not passed directly to ultralytics train().")

    def _normalize_config(self, cfg: dict) -> dict:
        training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
        augmentation_cfg = cfg.get("augmentation", {}) if isinstance(cfg.get("augmentation"), dict) else {}

        training_map = {
            "epochs": "epochs",
            "patience": "patience",
            "batch_size": "batch",
            "img_size": "imgsz",
            "learning_rate": "lr0",
            "lrf": "lrf",
            "optimizer": "optimizer",
            "weight_decay": "weight_decay",
            "warmup_epochs": "warmup_epochs",
            "warmup_momentum": "warmup_momentum",
            "box_loss_gain": "box",
            "cls_loss_gain": "cls",
            "dfl_loss_gain": "dfl",
            "freeze": "freeze",
        }

        augmentation_map = {
            "hsv_h": "hsv_h",
            "hsv_s": "hsv_s",
            "hsv_v": "hsv_v",
            "degrees": "degrees",
            "translate": "translate",
            "scale": "scale",
            "shear": "shear",
            "perspective": "perspective",
            "flipud": "flipud",
            "fliplr": "fliplr",
            "mosaic": "mosaic",
            "mixup": "mixup",
            "copy_paste": "copy_paste",
            "auto_augment": "auto_augment",
        }

        for src_key, dest_key in training_map.items():
            if dest_key not in cfg and src_key in training_cfg:
                cfg[dest_key] = training_cfg[src_key]

        for src_key, dest_key in augmentation_map.items():
            if dest_key not in cfg and src_key in augmentation_cfg:
                cfg[dest_key] = augmentation_cfg[src_key]

        return cfg

    @staticmethod
    def load_config(config_path: Path) -> dict:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _resolve_dataset_yaml(self, cfg: dict) -> str:
        data_cfg = cfg.get("data")
        if isinstance(data_cfg, str):
            return data_cfg

        if isinstance(data_cfg, dict):
            data_path = data_cfg.get("path", "underwater_dataset")
            names = data_cfg.get("class_names") or data_cfg.get("names") or []
            nc = data_cfg.get("num_classes") or data_cfg.get("nc") or len(names)
            class_weights = data_cfg.get("class_weights") or cfg.get("cls_weights")

            dataset_config = {
                "path": data_path,
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": int(nc) if nc else len(names),
                "names": names,
            }
            if class_weights:
                dataset_config["class_weights"] = class_weights

            yaml_path = self.config_path.with_name("dataset_optimized.yaml")
            with yaml_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(dataset_config, handle, sort_keys=False)
            return str(yaml_path)

        return "underwater_dataset/dataset.yaml"

    def _build_train_args(self, run_name: str, project_dir: str) -> dict:
        cfg = self._normalize_config(dict(self.config))
        cfg["data"] = self._resolve_dataset_yaml(cfg)
        cfg.setdefault("data", "underwater_dataset/dataset.yaml")
        cfg.setdefault("epochs", 100)
        cfg.setdefault("patience", 20)
        cfg.setdefault("batch", 8)
        cfg.setdefault("imgsz", 640)
        cfg.setdefault("lr0", 0.0005)
        cfg.setdefault("lrf", 0.01)
        cfg.setdefault("optimizer", "AdamW")
        cfg.setdefault("weight_decay", 0.0005)
        cfg.setdefault("freeze", 10)
        cfg.setdefault("mosaic", 1.0)
        cfg.setdefault("copy_paste", 0.5)
        cfg.setdefault("mixup", 0.5)
        cfg.setdefault("box", 7.5)
        cfg.setdefault("cls", 0.5)
        cfg.setdefault("dfl", 1.5)
        cfg.setdefault("warmup_epochs", 3)
        cfg.setdefault("warmup_momentum", 0.8)

        cfg.update(
            {
                "device": self.device,
                "name": run_name,
                "project": project_dir,
                "exist_ok": True,
                "verbose": True,
                "plots": True,
                "workers": 4,
            }
        )

        return {k: v for k, v in cfg.items() if k in SUPPORTED_TRAIN_KEYS}

    @staticmethod
    def _resolve_results_dir(results, project_dir: str, run_name: str) -> Path:
        if hasattr(results, "save_dir"):
            return Path(str(results.save_dir))
        if hasattr(results, "trainer") and hasattr(results.trainer, "save_dir"):
            return Path(str(results.trainer.save_dir))
        return Path(project_dir) / run_name

    @staticmethod
    def _plot_training_curves(results_dir: Path) -> None:
        csv_path = results_dir / "results.csv"
        if not csv_path.exists():
            print(f"Training curves skipped: missing {csv_path}")
            return

        df = pd.read_csv(csv_path)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="Box Loss")
        axes[0, 0].plot(df["epoch"], df["train/cls_loss"], label="Class Loss")
        axes[0, 0].plot(df["epoch"], df["train/dfl_loss"], label="DFL Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="green")
        axes[0, 1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="blue")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("mAP")
        axes[0, 1].set_title("Mean Average Precision")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="orange")
        axes[1, 0].plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="purple")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Score")
        axes[1, 0].set_title("Precision and Recall")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        if "lr/pg0" in df.columns:
            axes[1, 1].plot(df["epoch"], df["lr/pg0"], label="Learning Rate", color="red")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("LR")
            axes[1, 1].set_title("Learning Rate Schedule")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(results_dir / "training_curves.png", dpi=150)
        plt.close()

    def fine_tune(self, run_name: str = "underwater_finetuned", project_dir: str = "runs/finetune"):
        model = YOLO(self.base_model)
        train_args = self._build_train_args(run_name=run_name, project_dir=project_dir)

        print("Fine-tuning configuration:")
        for key in sorted(train_args):
            print(f"  {key}: {train_args[key]}")

        results = model.train(**train_args)

        results_dir = self._resolve_results_dir(results, project_dir, run_name)
        best_model = results_dir / "weights" / "best.pt"
        print("Fine-tuning complete")
        print(f"Best model path: {best_model}")

        self._plot_training_curves(results_dir)
        metrics_csv = results_dir / "results.csv"
        if metrics_csv.exists():
            metrics_copy = results_dir / "training_metrics.csv"
            if not metrics_copy.exists():
                metrics_copy.write_text(metrics_csv.read_text(encoding="utf-8"), encoding="utf-8")

        return model, results

    def run_progressive(
        self,
        run_name: str = "underwater_finetuned_progressive",
        project_dir: str = "runs/finetune",
        stage1_epochs: int = 30,
        stage2_epochs: int = 30,
        stage3_epochs: int = 50,
    ):
        model = YOLO(self.base_model)

        print("Progressive fine-tuning: Stage 1 (head-focused)")
        model.train(
            data=self.config.get("data", "underwater_dataset/dataset.yaml"),
            freeze=20,
            lr0=0.001,
            epochs=stage1_epochs,
            batch=self.config.get("batch", 8),
            imgsz=self.config.get("imgsz", 640),
            project=project_dir,
            name=f"{run_name}_stage1",
            exist_ok=True,
            device=self.device,
            verbose=True,
            plots=True,
        )

        print("Progressive fine-tuning: Stage 2 (partial unfreeze)")
        model.train(
            data=self.config.get("data", "underwater_dataset/dataset.yaml"),
            freeze=10,
            lr0=0.0005,
            epochs=stage2_epochs,
            batch=self.config.get("batch", 8),
            imgsz=self.config.get("imgsz", 640),
            project=project_dir,
            name=f"{run_name}_stage2",
            exist_ok=True,
            device=self.device,
            verbose=True,
            plots=True,
        )

        print("Progressive fine-tuning: Stage 3 (full model)")
        model.train(
            data=self.config.get("data", "underwater_dataset/dataset.yaml"),
            freeze=0,
            lr0=0.0001,
            epochs=stage3_epochs,
            batch=self.config.get("batch", 8),
            imgsz=self.config.get("imgsz", 640),
            project=project_dir,
            name=f"{run_name}_stage3",
            exist_ok=True,
            device=self.device,
            verbose=True,
            plots=True,
        )

        best_model = Path(project_dir) / f"{run_name}_stage3" / "weights" / "best.pt"
        print("Progressive fine-tuning complete")
        print(f"Best model path: {best_model}")
        return best_model

    def validate(self, model_path: str | None = None):
        if model_path is None:
            model_path = "runs/finetune/underwater_finetuned/weights/best.pt"

        model = YOLO(model_path)
        results = model.val(data=self.config.get("data", "underwater_dataset/dataset.yaml"))

        print("Validation Results:")
        print(f"mAP50:     {results.box.map50:.4f}")
        print(f"mAP50-95:  {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall:    {results.box.mr:.4f}")

        return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for underwater threat detection")
    parser.add_argument("--config", default="finetune_config.yaml", help="Fine-tuning YAML config")
    parser.add_argument("--model", default=None, help="Override base model, e.g. yolov8m.pt")
    parser.add_argument("--data", default=None, help="Override dataset yaml path")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--freeze", type=int, default=None, help="Override freeze layers")
    parser.add_argument("--lr0", type=float, default=None, help="Override initial LR")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--run-name", default="underwater_finetuned", help="YOLO run name")
    parser.add_argument("--project", default="runs/finetune", help="YOLO output project")
    parser.add_argument("--progressive", action="store_true", help="Run 3-stage progressive fine-tuning")
    parser.add_argument("--validate-only", action="store_true", help="Skip training and only validate")
    parser.add_argument("--model-path", default=None, help="Model path for validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if args.model is not None:
        cfg["model"] = args.model
    if args.data is not None:
        cfg["data"] = args.data
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.freeze is not None:
        cfg["freeze"] = int(args.freeze)
    if args.lr0 is not None:
        cfg["lr0"] = float(args.lr0)
    if args.batch is not None:
        cfg["batch"] = int(args.batch)

    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    tuner = UnderwaterFineTuner(
        base_model=cfg.get("model", "yolov8m.pt"),
        config_path=str(cfg_path),
    )

    if args.validate_only:
        if args.model_path:
            tuner.validate(model_path=args.model_path)
        else:
            print("Validation skipped: --model-path is required when using --validate-only.")
        return

    if args.progressive:
        tuner.run_progressive(run_name=args.run_name, project_dir=args.project)
        if args.model_path:
            tuner.validate(model_path=args.model_path)
        else:
            print("Validation skipped: --model-path was not provided.")
        return

    tuner.fine_tune(run_name=args.run_name, project_dir=args.project)
    if args.model_path:
        tuner.validate(model_path=args.model_path)
    else:
        print("Validation skipped: --model-path was not provided.")


if __name__ == "__main__":
    main()
