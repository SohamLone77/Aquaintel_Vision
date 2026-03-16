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


def main(config=None):
    """
    Launch training with optional config overrides.

    Args:
        config: dict of training parameters (see train_complete.py for keys).
                Defaults optimised for the full 890-pair dataset.
    """
    default_config = {
        'data_path': 'data',
        'img_size': 128,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 1e-4,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'use_tensorboard': True,
        'use_csv_logger': True,
    }

    if config:
        default_config.update(config)

    trainer = UnderwaterTrainer(default_config)
    trainer.train()
    trainer.evaluate()
    trainer.predict_sample()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
