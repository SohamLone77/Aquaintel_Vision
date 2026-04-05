"""Experiment tracking and leaderboard utilities for U-Net and YOLO."""

from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


class ExperimentTracker:
    def __init__(self, leaderboard_file="experiments/leaderboard.json"):
        self.leaderboard_file = leaderboard_file
        os.makedirs(os.path.dirname(leaderboard_file), exist_ok=True)
        self.load_leaderboard()

    def load_leaderboard(self):
        if os.path.exists(self.leaderboard_file):
            try:
                with open(self.leaderboard_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                data = None

            if isinstance(data, dict):
                self.leaderboard = {
                    "unet": data.get("unet", []) if isinstance(data.get("unet", []), list) else [],
                    "yolo": data.get("yolo", []) if isinstance(data.get("yolo", []), list) else [],
                }
            else:
                self.leaderboard = {"unet": [], "yolo": []}
        else:
            self.leaderboard = {"unet": [], "yolo": []}

    def save_leaderboard(self):
        with open(self.leaderboard_file, "w", encoding="utf-8") as f:
            json.dump(self.leaderboard, f, indent=2)

    def check_unet_gates(self, metrics):
        val_loss = metrics.get("val_loss")
        val_mae = metrics.get("val_mae")
        ssim = metrics.get("ssim")
        gates = {
            "val_loss": (val_loss is not None and val_loss <= 0.08),
            "val_mae": (val_mae is not None and val_mae <= 0.09),
            "ssim": (ssim is not None and ssim >= 0.85),
        }
        return {"passed": all(gates.values()), "details": gates}

    def check_yolo_gates(self, metrics):
        gates = {
            "mAP50": metrics.get("mAP50", 0) >= 0.75,
            "precision": metrics.get("precision", 0) >= 0.70,
            "recall": metrics.get("recall", 0) >= 0.65,
        }
        return {"passed": all(gates.values()), "details": gates}

    def register_unet_experiment(self, run_id, metrics, model_path, is_promoted=False):
        acceptance = self.check_unet_gates(metrics)
        exp = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "model_path": model_path,
            "is_promoted": bool(is_promoted),
            "accepted": acceptance["passed"],
            "acceptance_details": acceptance["details"],
        }
        self.leaderboard["unet"].append(exp)
        self.leaderboard["unet"] = sorted(self.leaderboard["unet"], key=lambda x: x["metrics"].get("val_loss") or 9999)[:50]
        self.save_leaderboard()
        return exp

    def register_yolo_experiment(self, run_id, metrics, model_path, is_promoted=False):
        acceptance = self.check_yolo_gates(metrics)
        exp = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "model_path": model_path,
            "is_promoted": bool(is_promoted),
            "accepted": acceptance["passed"],
            "acceptance_details": acceptance["details"],
        }
        self.leaderboard["yolo"].append(exp)
        self.leaderboard["yolo"] = sorted(self.leaderboard["yolo"], key=lambda x: -(x["metrics"].get("mAP50") or 0))[:50]
        self.save_leaderboard()
        return exp

    def generate_leaderboard_table(self):
        unet_df = pd.DataFrame([
            {
                "Run ID": e["run_id"],
                "Val Loss": e["metrics"].get("val_loss"),
                "Val MAE": e["metrics"].get("val_mae"),
                "SSIM": e["metrics"].get("ssim"),
                "Accepted": e.get("accepted", False),
                "Promoted": e.get("is_promoted", False),
            }
            for e in self.leaderboard["unet"]
        ])
        yolo_df = pd.DataFrame([
            {
                "Run ID": e["run_id"],
                "mAP50": e["metrics"].get("mAP50"),
                "Precision": e["metrics"].get("precision"),
                "Recall": e["metrics"].get("recall"),
                "Accepted": e.get("accepted", False),
                "Promoted": e.get("is_promoted", False),
            }
            for e in self.leaderboard["yolo"]
        ])
        return unet_df, yolo_df

    def plot_comparison(self, output_path="experiments/comparison_plot.png"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if self.leaderboard["unet"]:
            runs = [e["run_id"][:12] for e in self.leaderboard["unet"][:10]]
            losses = [e["metrics"].get("val_loss", 0) or 0 for e in self.leaderboard["unet"][:10]]
            axes[0].barh(runs, losses)
            axes[0].set_title("U-Net (lower is better)")

        if self.leaderboard["yolo"]:
            runs = [e["run_id"][:12] for e in self.leaderboard["yolo"][:10]]
            maps = [e["metrics"].get("mAP50", 0) or 0 for e in self.leaderboard["yolo"][:10]]
            axes[1].barh(runs, maps)
            axes[1].set_title("YOLO mAP50 (higher is better)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    tracker = ExperimentTracker()
    unet_df, yolo_df = tracker.generate_leaderboard_table()
    print("U-Net leaderboard")
    print(unet_df.to_string(index=False) if not unet_df.empty else "(empty)")
    print("YOLO leaderboard")
    print(yolo_df.to_string(index=False) if not yolo_df.empty else "(empty)")
