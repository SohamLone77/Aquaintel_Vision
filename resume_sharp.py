#!/usr/bin/env python3
"""Resume fine-tuning with sharpness-focused loss."""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(str(Path(__file__).parent))

from data_loader_simple import SimpleDataLoader
from losses.simple_losses import SimpleLosses


def find_latest_model():
    search_paths = [
        "models/checkpoints/*.h5",
        "models/checkpoints/*.keras",
        "models/*.h5",
    ]

    all_models = []
    for pattern in search_paths:
        all_models.extend(glob.glob(pattern))

    if not all_models:
        print("❌ No model files found")
        return None

    latest_model = max(all_models, key=os.path.getmtime)
    file_size = os.path.getsize(latest_model) / (1024 * 1024)
    mod_time = datetime.fromtimestamp(os.path.getmtime(latest_model))

    print(f"📂 Found model: {os.path.basename(latest_model)}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return latest_model


def plot_finetuning_history(history, timestamp):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Fine-tuning Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "mae" in history.history:
        axes[1].plot(history.history["mae"], label="Training MAE", linewidth=2)
        if "val_mae" in history.history:
            axes[1].plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Fine-tuning MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("Sharp Fine-tuning History", fontsize=14)
    plt.tight_layout()
    os.makedirs("results/training_plots", exist_ok=True)
    plt.savefig(f"results/training_plots/sharp_finetuned_{timestamp}_history.png", dpi=150)


def resume_sharp_training(additional_epochs=20):
    print("=" * 60)
    print("RESUME TRAINING FOR SHARPER RESULTS")
    print("=" * 60)

    model_path = find_latest_model()
    if not model_path:
        return None, None

    print(f"\n📂 Loading model: {os.path.basename(model_path)}")
    custom_objects = {
        "combined_loss": SimpleLosses.combined_loss,
        "sharp_loss": SimpleLosses.sharp_loss,
        "mse_loss": SimpleLosses.mse_loss,
        "mae_loss": SimpleLosses.mae_loss,
        "ssim_loss": SimpleLosses.ssim_loss,
        "edge_loss": SimpleLosses.edge_loss,
        "gradient_loss": SimpleLosses.gradient_loss,
    }

    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✅ Model loaded successfully")
    except Exception as exc:
        print(f"⚠️ Loading with custom objects failed: {exc}")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded without custom objects")

    model_input_size = int(model.input_shape[1])
    print(f"📐 Model input size detected: {model_input_size}x{model_input_size}")

    print("\n📂 Loading data with higher resolution...")
    loader = SimpleDataLoader(
        data_path="data",
        img_size=model_input_size,
        batch_size=2,
        validation_split=0.2,
    )

    train_dataset = loader.get_dataset("train")
    val_dataset = loader.get_dataset("validation")
    if train_dataset is None:
        print("❌ No training data available")
        return None, None

    print("\n⚙️ Recompiling with SHARP loss for fine-tuning...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=SimpleLosses.sharp_loss,
        metrics=["mae"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs/csv", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"models/checkpoints/sharp_finetuned_{timestamp}_best.h5",
            monitor="val_loss",
            save_best_only=True,
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
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        tf.keras.callbacks.CSVLogger(
            filename=f"logs/csv/sharp_finetuned_{timestamp}.csv",
            separator=",",
            append=False,
        ),
    ]

    print(f"\n🚀 Fine-tuning for {additional_epochs} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=additional_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = f"models/checkpoints/sharp_model_final_{timestamp}.h5"
    model.save(final_path)
    print(f"\n✅ Sharp model saved: {final_path}")

    plot_finetuning_history(history, timestamp)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume training for sharper results")
    parser.add_argument("--epochs", type=int, default=20, help="Additional epochs to train")
    args = parser.parse_args()

    resume_sharp_training(additional_epochs=args.epochs)
