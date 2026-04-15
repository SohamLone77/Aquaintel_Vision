"""Enhanced U-Net training launcher with robust logging and resume support."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from train_unet import run_training_pipeline
from utils.config_loader import load_runtime_config


def setup_logging(run_id: str):
    log_dir = Path("logs/training") / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"train_unet_enhanced.{run_id}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_dir


def find_latest_checkpoint(checkpoint_dir="models/checkpoints"):
    paths = sorted(Path(checkpoint_dir).glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(paths[0]) if paths else None


def main():
    parser = argparse.ArgumentParser(description="Enhanced U-Net training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--preserve-aspect", action="store_true")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_dir = setup_logging(run_id)

    overrides = {
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "model_name": f"unet_enhanced_{run_id}",
        "deterministic_mode": bool(args.deterministic),
        "preserve_aspect_ratio": bool(args.preserve_aspect),
    }

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = args.checkpoint or find_latest_checkpoint()
        if resume_checkpoint is None:
            raise SystemExit("Resume requested but no checkpoint was found.")
        logger.info("Resume requested. Checkpoint: %s", resume_checkpoint)

    cfg = load_runtime_config(overrides=overrides)
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    logger.info("Starting enhanced training run: %s", run_id)
    _, history = run_training_pipeline(cfg, resume_checkpoint=resume_checkpoint)

    summary = {
        "run_id": run_id,
        "epochs_ran": len(history.history.get("loss", [])) if history is not None else 0,
        "final_loss": history.history.get("loss", [None])[-1] if history is not None else None,
        "final_val_loss": history.history.get("val_loss", [None])[-1] if history is not None else None,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Training complete. Logs: %s", log_dir)


if __name__ == "__main__":
    main()
