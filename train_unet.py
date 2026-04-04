"""
Training script for U-Net underwater image enhancement.
Lightweight entry-point that delegates to UnderwaterTrainer in train_complete.py.
Run directly or import and call main() with a custom config dict.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_complete import UnderwaterTrainer  # noqa: E402
from utils.config_loader import ConfigError, load_runtime_config  # noqa: E402


def main(config=None):
    """
    Launch training with optional config overrides.

    Args:
        config: dict of training parameters (see train_complete.py for keys).
                Defaults optimised for the full 890-pair dataset.
    """
    try:
        runtime_config = load_runtime_config(overrides=config)
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    trainer = UnderwaterTrainer(runtime_config)
    trainer.train()
    trainer.evaluate()
    trainer.predict_sample()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
