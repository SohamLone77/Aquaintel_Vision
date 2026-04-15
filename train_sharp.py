#!/usr/bin/env python3
"""Training script for sharp underwater image enhancement."""

import os
import sys
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.basic_unet import build_basic_unet
from training.data_loader_simple import SimpleDataLoader
from losses.simple_losses import SimpleLosses
from utils.gpu import configure_tensorflow_device
from utils.model_registry import ModelRegistry

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("logs/csv", exist_ok=True)
os.makedirs("results/training_plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


class SharpTrainer:
    """Trainer with edge-preserving loss for sharper results."""

    def __init__(self, config=None):
        self.config = {
            "data_path": "data",
            "img_size": 128,
            "batch_size": 2,
            "epochs": 30,
            "learning_rate": 1e-4,
            "validation_split": 0.2,
            "model_name": f"sharp128_unet_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "loss_type": "sharp",
            "registry_path": "results/model_registry.json",
        }

        if config:
            self.config.update(config)

        print("=" * 60)
        print("SHARP UNDERWATER IMAGE ENHANCEMENT TRAINER")
        print("=" * 60)
        print("\n📋 Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")

        device_info = configure_tensorflow_device(self.config)
        print(
            f"\n🧠 TensorFlow device: {device_info['device']} "
            f"(GPUs: {device_info['gpu_count']}, mixed_precision: {device_info['mixed_precision']})"
        )

        self.setup_data()
        self.setup_model()

    def setup_data(self):
        print("\n📂 Loading data...")

        self.loader = SimpleDataLoader(
            data_path=self.config["data_path"],
            img_size=self.config["img_size"],
            batch_size=self.config["batch_size"],
            validation_split=self.config["validation_split"],
        )

        self.train_dataset = self.loader.get_dataset("train")
        self.val_dataset = self.loader.get_dataset("validation")

        if self.train_dataset is None:
            raise ValueError("No training data available")

        print("✅ Data loaded successfully")

    def setup_model(self):
        print("\n🏗️ Building model...")
        self.model = build_basic_unet(input_shape=(self.config["img_size"], self.config["img_size"], 3))

        if self.config["loss_type"] == "sharp":
            loss_fn = SimpleLosses.sharp_loss
            print("✅ Using SHARP loss (40% MSE, 40% Edge, 20% Gradient)")
        else:
            loss_fn = SimpleLosses.combined_loss
            print("✅ Using COMBINED loss")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss=loss_fn,
            metrics=["mae"],
        )

        print(f"✅ Model built with {self.model.count_params():,} parameters")

        summary_path = f"results/{self.config['model_name']}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

    def train(self):
        print("\n🚀 Starting training...")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f"models/checkpoints/{self.config['model_name']}_best.h5",
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
                mode="min",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode="min",
            ),
            tf.keras.callbacks.CSVLogger(
                filename=f"logs/csv/{self.config['model_name']}.csv",
                separator=",",
                append=False,
            ),
        ]

        try:
            if find_spec("tensorboard") is None:
                raise ImportError("tensorboard not installed")
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=f"logs/{self.config['model_name']}",
                    histogram_freq=1,
                    write_graph=True,
                    update_freq="epoch",
                )
            )
        except Exception:
            print("⚠️ TensorBoard unavailable; continuing without it.")

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config["epochs"],
            callbacks=callbacks,
            verbose=1,
        )

        print("\n✅ Training complete!")

        final_path_h5 = f"models/checkpoints/{self.config['model_name']}_final.h5"
        final_path_keras = f"models/checkpoints/{self.config['model_name']}_final.keras"
        self.model.save(final_path_h5)
        self.model.save(final_path_keras)

        print(f"💾 Final model saved (H5): {final_path_h5}")
        print(f"💾 Final model saved (Keras): {final_path_keras}")

        history_plot = self.save_training_history(history)
        self.record_training_metadata(history, final_path_h5, final_path_keras, history_plot)
        return history

    def save_training_history(self, history):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2)
        if "val_loss" in history.history:
            axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "mae" in history.history:
            axes[1].plot(history.history["mae"], label="Training MAE", linewidth=2)
            if "val_mae" in history.history:
                axes[1].plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MAE")
            axes[1].set_title("Mean Absolute Error")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"Training History - {self.config['model_name']}", fontsize=14)
        plt.tight_layout()

        save_path = f"results/training_plots/{self.config['model_name']}_history.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Training history saved: {save_path}")
        return save_path

    def record_training_metadata(self, history, final_h5, final_keras, history_plot):
        metrics = {
            "final_loss": float(history.history["loss"][-1]),
            "final_mae": float(history.history.get("mae", [0])[-1]),
            "final_val_loss": float(history.history.get("val_loss", [0])[-1]),
            "final_val_mae": float(history.history.get("val_mae", [0])[-1]),
            "epochs_ran": len(history.epoch),
        }

        artifacts = {
            "best_checkpoint": f"models/checkpoints/{self.config['model_name']}_best.h5",
            "final_h5": final_h5,
            "final_keras": final_keras,
            "history_plot": history_plot,
        }

        registry = ModelRegistry(self.config["registry_path"])
        registry.register_training_run(
            run_name=self.config["model_name"],
            config=self.config,
            metrics=metrics,
            artifacts=artifacts,
        )
        print(f"🗂️ Registered run metadata in {self.config['registry_path']}")


def main():
    config = {
        "data_path": "data",
        "img_size": 128,
        "batch_size": 2,
        "epochs": 30,
        "learning_rate": 1e-4,
        "loss_type": "sharp",
    }

    trainer = SharpTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
