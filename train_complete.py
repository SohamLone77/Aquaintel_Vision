#!/usr/bin/env python3
"""
Complete training script for U-Net underwater image enhancement
with all fixes and error handling
"""

import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so package imports work regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

print("=" * 60)
print("UNDERWATER IMAGE ENHANCEMENT - TRAINING")
print("=" * 60)

# Ensure required directories exist
for dir_name in ['models/checkpoints', 'losses', 'training',
                 'logs', 'logs/csv',
                 'results/training_plots']:
    os.makedirs(dir_name, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from models.basic_unet import build_basic_unet                      # noqa: E402
except ModuleNotFoundError:
    from basic_unet import build_basic_unet                             # noqa: E402
from losses.simple_losses import SimpleLosses                            # noqa: E402
from losses.underwater_losses import (UnderwaterLosses,                  # noqa: E402
                                       CombinedUnderwaterLoss,
                                       create_loss_function)
from training.data_loader import UnderwaterDataLoader                    # noqa: E402
from training.callbacks import create_all_callbacks, CustomCallback      # noqa: E402
from utils.config_loader import ConfigError, load_runtime_config         # noqa: E402
from utils.model_registry import ModelRegistry                            # noqa: E402
from scripts.validate_dataset import validate_dataset                     # noqa: E402

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt                                          # noqa: E402
from datetime import datetime                                            # noqa: E402

print("✅ All modules imported successfully")

class UnderwaterTrainer:
    """
    Trainer class for underwater image enhancement
    """
    
    def __init__(self, config=None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Dictionary with training parameters
        """
        if config is None:
            self.config = load_runtime_config()
        else:
            self.config = config

        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['results_dir'], 'training_plots'), exist_ok=True)
        
        print("\n📋 Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        
        # Initialize data loader
        self.setup_data()
        
        # Build model
        self.setup_model()
        
        # Setup callbacks
        self.setup_callbacks()
    
    def setup_data(self):
        """Initialize data loader and datasets"""
        print("\n📂 Loading data...")

        is_valid, details = validate_dataset(data_path=self.config['data_path'])
        if not is_valid:
            issues = "\n - ".join(details['issues'])
            raise RuntimeError(
                f"Dataset validation failed for {self.config['data_path']}:\n - {issues}"
            )
        
        self.loader = UnderwaterDataLoader(
            data_path=self.config['data_path'],
            img_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            augment=self.config.get('augment_enabled', True),
            augmentation_config={
                'profile': self.config.get('augmentation_profile', 'standard'),
                **self.config.get('augmentation', {}),
            }
        )
        
        self.train_dataset = self.loader.get_dataset('train')
        self.val_dataset = self.loader.get_dataset('validation')
        
        if self.train_dataset is None:
            raise RuntimeError("Dataset loader returned no training dataset.")
        
        # Print dataset info
        if hasattr(self.loader, 'train_indices'):
            print(f"✅ Training samples: {len(self.loader.train_indices)}")
        if hasattr(self.loader, 'val_indices') and self.val_dataset is not None:
            print(f"✅ Validation samples: {len(self.loader.val_indices)}")
    
    def _create_dummy_data(self):
        """Create dummy data for testing"""
        print("Creating dummy dataset for testing...")
        
        # Create dummy numpy data
        x_dummy = np.random.rand(10, self.config['img_size'], self.config['img_size'], 3).astype(np.float32)
        y_dummy = np.random.rand(10, self.config['img_size'], self.config['img_size'], 3).astype(np.float32)
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices((x_dummy, y_dummy))
        self.train_dataset = self.train_dataset.batch(self.config['batch_size'])
        self.val_dataset = self.train_dataset.take(1)
        
        print("✅ Created dummy dataset")
    
    def setup_model(self):
        """Build and compile model"""
        print("\n🏗️ Building model...")
        
        # Build model
        self.model = build_basic_unet(
            input_shape=(self.config['img_size'], self.config['img_size'], 3)
        )
        
        # Configure and compile model loss
        loss_type = str(self.config.get('loss_type', 'combined')).lower()
        if loss_type == 'combined':
            SimpleLosses.combined_alpha = float(self.config.get('ssim_weight', 0.5))
            selected_loss = SimpleLosses.combined_loss
            print(f"🔧 Loss: combined (ssim_weight={SimpleLosses.combined_alpha})")
        elif loss_type == 'mse':
            selected_loss = SimpleLosses.mse_loss
            print("🔧 Loss: mse")
        elif loss_type == 'mae':
            selected_loss = SimpleLosses.mae_loss
            print("🔧 Loss: mae")
        elif loss_type == 'ssim':
            selected_loss = SimpleLosses.ssim_loss
            print("🔧 Loss: ssim")
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=selected_loss,
            metrics=['mae']
        )
        
        print(f"✅ Model built with {self.model.count_params():,} parameters")
        
        # Save model summary
        summary_path = os.path.join(
            self.config['results_dir'],
            f"{self.config['model_name']}_summary.txt"
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        print("\n📞 Setting up callbacks...")
        
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.config['checkpoint_dir'], f"{self.config['model_name']}_best.h5"),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if self.config['use_tensorboard']:
            try:
                import tensorboard  # noqa: F401
                log_dir = f"logs/{self.config['model_name']}"
                self.callbacks.append(
                    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                )
            except Exception:
                print("⚠️ TensorBoard unavailable; continuing without TensorBoard callback.")
        
        # CSV logger for easy progress monitoring
        if self.config['use_csv_logger']:
            csv_log_path = f"logs/csv/{self.config['model_name']}_training.csv"
            os.makedirs("logs/csv", exist_ok=True)
            self.callbacks.append(tf.keras.callbacks.CSVLogger(csv_log_path))
        
        print(f"✅ Created {len(self.callbacks)} callbacks")
    
    def train(self):
        """Run training"""
        print("\n🚀 Starting training...")
        
        # Compute steps per epoch
        steps_per_epoch = self.loader.train_steps
        validation_steps = self.loader.val_steps if self.val_dataset else None
        
        print(f"   Steps per epoch: {steps_per_epoch}")
        if validation_steps:
            print(f"   Validation steps: {validation_steps}")
        
        # Train model
        try:
            self.history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=self.config['epochs'],
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=self.callbacks,
                verbose=1
            )
        except Exception as exc:
            if "TensorBoard is not installed" in str(exc) or "TBNotInstalledError" in type(exc).__name__:
                print("⚠️ TensorBoard runtime unavailable; retrying training without TensorBoard callback.")
                self.callbacks = [
                    callback for callback in self.callbacks
                    if not isinstance(callback, tf.keras.callbacks.TensorBoard)
                ]
                self.history = self.model.fit(
                    self.train_dataset,
                    validation_data=self.val_dataset,
                    epochs=self.config['epochs'],
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    callbacks=self.callbacks,
                    verbose=1
                )
            else:
                raise
        
        print("\n✅ Training complete!")
        
        # Plot training history
        self.plot_history()
        
        # Save final model in both formats
        final_h5 = os.path.join(self.config['checkpoint_dir'], f"{self.config['model_name']}_final.h5")
        final_keras = os.path.join(self.config['checkpoint_dir'], f"{self.config['model_name']}_final.keras")
        self.model.save(final_h5)
        self.model.save(final_keras)
        print(f"💾 Final model saved to: {final_h5}")
        print(f"💾 Final model saved to: {final_keras}")

        self._record_training_metadata(final_h5, final_keras)
        
        return self.history

    def _record_training_metadata(self, final_h5, final_keras):
        """Record a training run and its artifacts in the model registry."""
        metrics = {
            'final_loss': self.history.history['loss'][-1],
            'final_mae': self.history.history.get('mae', [None])[-1],
            'final_val_loss': self.history.history.get('val_loss', [None])[-1],
            'final_val_mae': self.history.history.get('val_mae', [None])[-1],
            'epochs_ran': len(self.history.epoch),
        }

        artifacts = {
            'best_checkpoint': os.path.join(
                self.config['checkpoint_dir'],
                f"{self.config['model_name']}_best.h5"
            ),
            'final_h5': final_h5,
            'final_keras': final_keras,
            'history_plot': os.path.join(
                self.config['results_dir'],
                'training_plots',
                f"{self.config['model_name']}_history.png"
            ),
            'sample_plot': os.path.join(
                self.config['results_dir'],
                'training_plots',
                f"{self.config['model_name']}_samples.png"
            ),
        }

        registry = ModelRegistry(self.config['registry_path'])
        registry.register_training_run(
            run_name=self.config['model_name'],
            config=self.config,
            metrics=metrics,
            artifacts=artifacts,
        )
    
    def plot_history(self):
        """Plot training history"""
        if not hasattr(self, 'history'):
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        if 'mae' in self.history.history:
            axes[1].plot(self.history.history['mae'], label='Train MAE')
            if 'val_mae' in self.history.history:
                axes[1].plot(self.history.history['val_mae'], label='Val MAE')
            axes[1].set_title('Mean Absolute Error')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        history_plot_path = os.path.join(
            self.config['results_dir'],
            'training_plots',
            f"{self.config['model_name']}_history.png"
        )
        plt.savefig(history_plot_path, dpi=150)
        plt.close(fig)
        
        print(f"📊 Training history saved to {history_plot_path}")
    
    def evaluate(self):
        """Evaluate model on validation data"""
        if self.val_dataset is None:
            print("No validation data available")
            return
        
        print("\n📊 Evaluating model...")
        results = self.model.evaluate(
            self.val_dataset,
            steps=self.loader.val_steps,
            verbose=1
        )
        
        print("\n📈 Evaluation Results:")
        metric_names = ['loss', 'mae']
        if not isinstance(results, list):
            results = [results]
        for i, val in enumerate(results):
            name = metric_names[i] if i < len(metric_names) else f'metric_{i}'
            print(f"   {name}: {val:.4f}")
        
        return results
    
    def predict_sample(self, n_samples=4):
        """Predict on sample images"""
        if self.val_dataset is None:
            print("No validation data for predictions")
            return
        
        print("\n🖼️ Generating sample predictions...")
        
        # Get a batch
        for batch_x, batch_y in self.val_dataset.take(1):
            predictions = self.model.predict(batch_x, verbose=0)
            
            # Plot
            fig, axes = plt.subplots(min(n_samples, len(batch_x)), 3, 
                                    figsize=(12, 4*min(n_samples, len(batch_x))))
            
            if n_samples == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(min(n_samples, len(batch_x))):
                axes[i, 0].imshow(batch_x[i])
                axes[i, 0].set_title('Input')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(predictions[i])
                axes[i, 1].set_title('Prediction')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(batch_y[i])
                axes[i, 2].set_title('Target')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            sample_plot_path = os.path.join(
                self.config['results_dir'],
                'training_plots',
                f"{self.config['model_name']}_samples.png"
            )
            plt.savefig(sample_plot_path, dpi=150)
            plt.close(fig)
            
            print(f"✅ Sample predictions saved")

def main():
    """Main training function"""

    try:
        config = load_runtime_config()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    # Create trainer
    trainer = UnderwaterTrainer(config)
    
    # Train model
    history = trainer.train()
    
    # Evaluate
    trainer.evaluate()
    
    # Generate predictions
    trainer.predict_sample()
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()