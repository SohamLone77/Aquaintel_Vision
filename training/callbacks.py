"""
Training callbacks for model training
"""

import tensorflow as tf
import numpy as np
import os
import datetime


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Custom callback for monitoring underwater image enhancement training.
    Logs per-epoch metrics and optionally saves sample predictions.
    """

    def __init__(self, log_dir="logs", save_samples=False, val_dataset=None):
        super().__init__()
        self.log_dir = log_dir
        self.save_samples = save_samples
        self.val_dataset = val_dataset
        self.best_val_loss = np.inf
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        val_loss = logs.get('val_loss', None)

        parts = [f"Epoch {epoch + 1}"]
        parts.append(f"lr={lr:.2e}")
        for k, v in logs.items():
            parts.append(f"{k}={v:.4f}")
        print(" | ".join(parts))

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            print(f"  ↳ New best val_loss: {val_loss:.4f}")

    def on_train_begin(self, logs=None):
        print("\n" + "=" * 50)
        print("Training started")
        print("=" * 50)

    def on_train_end(self, logs=None):
        print("\n" + "=" * 50)
        print("Training finished")
        if self.best_val_loss < np.inf:
            print(f"Best val_loss: {self.best_val_loss:.4f}")
        print("=" * 50)


class TrainingCallbacks:
    """Collection of training callbacks"""

    @staticmethod
    def get_checkpoint_callback(model_name="underwater_model"):
        checkpoint_dir = "models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filepath = os.path.join(checkpoint_dir, f"{model_name}_{timestamp}.h5")

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

    @staticmethod
    def get_early_stopping_callback(patience=10):
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

    @staticmethod
    def get_reduce_lr_callback(patience=5, factor=0.5):
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience,
            min_lr=1e-7,
            verbose=1
        )

    @staticmethod
    def get_csv_logger_callback(model_name="underwater_model"):
        log_dir = "logs/csv"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        filepath = os.path.join(log_dir, f"{model_name}_{timestamp}.csv")
        return tf.keras.callbacks.CSVLogger(filepath)


def create_all_callbacks(model_name="underwater_model",
                         early_stopping_patience=10,
                         reduce_lr_patience=5,
                         use_tensorboard=False,
                         use_csv_logger=True,
                         **kwargs):
    """Create all standard callbacks"""
    callbacks = [
        TrainingCallbacks.get_checkpoint_callback(model_name),
        TrainingCallbacks.get_early_stopping_callback(early_stopping_patience),
        TrainingCallbacks.get_reduce_lr_callback(patience=reduce_lr_patience),
        CustomCallback(),
    ]

    if use_csv_logger:
        callbacks.append(TrainingCallbacks.get_csv_logger_callback(model_name))

    if use_tensorboard:
        log_dir = os.path.join("logs", model_name)
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    return callbacks
