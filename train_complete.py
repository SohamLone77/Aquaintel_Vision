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
                 'data/raw', 'data/reference', 'logs', 'logs/csv',
                 'results/training_plots']:
    os.makedirs(dir_name, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from models.basic_unet import build_basic_unet                          # noqa: E402
from losses.simple_losses import SimpleLosses                            # noqa: E402
from losses.underwater_losses import (UnderwaterLosses,                  # noqa: E402
                                       CombinedUnderwaterLoss,
                                       create_loss_function)
from training.data_loader import UnderwaterDataLoader                    # noqa: E402
from training.callbacks import create_all_callbacks, CustomCallback      # noqa: E402

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
        # Default configuration
        self.config = {
            'data_path': 'data',
            'img_size': 128,
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 1e-4,
            'validation_split': 0.2,
            'model_name': f'unet_{datetime.now().strftime("%Y%m%d_%H%M")}',
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'use_tensorboard': True,
            'use_csv_logger': True
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
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
        
        self.loader = UnderwaterDataLoader(
            data_path=self.config['data_path'],
            img_size=self.config['img_size'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            augment=True
        )
        
        self.train_dataset = self.loader.get_dataset('train')
        self.val_dataset = self.loader.get_dataset('validation')
        
        if self.train_dataset is None:
            print("⚠️ No training data available! Creating dummy data...")
            self._create_dummy_data()
        
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
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss=SimpleLosses.combined_loss,
            metrics=['mae']
        )
        
        print(f"✅ Model built with {self.model.count_params():,} parameters")
        
        # Save model summary
        with open(f"results/{self.config['model_name']}_summary.txt", 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    def setup_callbacks(self):
        """Setup training callbacks"""
        print("\n📞 Setting up callbacks...")
        
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f"models/checkpoints/{self.config['model_name']}_best.h5",
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
            log_dir = f"logs/{self.config['model_name']}"
            self.callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            )
        
        # CSV logger for easy progress monitoring
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
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.config['epochs'],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=self.callbacks,
            verbose=1
        )
        
        print("\n✅ Training complete!")
        
        # Plot training history
        self.plot_history()
        
        # Save final model in both formats
        final_h5 = f"models/checkpoints/{self.config['model_name']}_final.h5"
        final_keras = f"models/checkpoints/{self.config['model_name']}_final.keras"
        self.model.save(final_h5)
        self.model.save(final_keras)
        print(f"💾 Final model saved to: {final_h5}")
        print(f"💾 Final model saved to: {final_keras}")
        
        return self.history
    
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
        plt.savefig(f"results/training_plots/{self.config['model_name']}_history.png", dpi=150)
        plt.close(fig)
        
        print(f"📊 Training history saved to results/training_plots/{self.config['model_name']}_history.png")
    
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
            plt.savefig(f"results/training_plots/{self.config['model_name']}_samples.png", dpi=150)
            plt.close(fig)
            
            print(f"✅ Sample predictions saved")

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_path': 'data',
        'img_size': 128,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 1e-4,
        'validation_split': 0.2,
    }
    
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