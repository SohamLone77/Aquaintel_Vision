"""Compatibility YOLO entrypoint that delegates to canonical train_yolo implementation."""

from __future__ import annotations

import argparse
from datetime import datetime

from train_yolo import UnderwaterYOLOTrainer


class StandardizedYOLOTrainer:
    def __init__(self, dataset_yaml="underwater_dataset/dataset.yaml", project_dir="runs/yolo", run_name=None, model_name="yolov8n.pt"):
        self.dataset_yaml = dataset_yaml
        self.project_dir = project_dir
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = model_name

    def train(self, epochs=100, imgsz=640, batch=16, device="cpu", allow_placeholder_labels=False):
        trainer = UnderwaterYOLOTrainer(
            dataset_yaml=self.dataset_yaml,
            model_name=self.model_name,
            run_name=self.run_name,
            project_dir=self.project_dir,
            allow_placeholder_labels=bool(allow_placeholder_labels),
        )
        if str(device).lower() == "cpu":
            trainer.device = "cpu"
        return trainer.train(epochs=epochs, imgsz=imgsz, batch=batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="underwater_dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project", type=str, default="runs/yolo")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--allow-placeholder-labels", action="store_true")
    args = parser.parse_args()

    trainer = StandardizedYOLOTrainer(
        dataset_yaml=args.data,
        project_dir=args.project,
        run_name=args.run_name,
        model_name=args.model,
    )
    trainer.train(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        allow_placeholder_labels=bool(args.allow_placeholder_labels),
    )


if __name__ == "__main__":
    main()
