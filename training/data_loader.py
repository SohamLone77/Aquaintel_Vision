"""
Enhanced Underwater Data Loader with more features
"""

import math
import glob
import os
import random

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

class UnderwaterDataLoader:
    def __init__(self, data_path="data", img_size=128, batch_size=8,
                 validation_split=0.2, augment=True, augmentation_config=None):
        """
        Enhanced data loader for underwater images
        
        Args:
            data_path: Path to data folder
            img_size: Target image size (square)
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            augment: Whether to apply data augmentation
        """
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.augment = augment
        self.augmentation_config = self._resolve_augmentation_config(augmentation_config)
        
        self.raw_folder = os.path.join(data_path, "raw")
        self.ref_folder = os.path.join(data_path, "reference")
        
        # Check if directories exist
        if not os.path.exists(self.raw_folder):
            print(f"⚠️ Raw folder not found: {self.raw_folder}")
            os.makedirs(self.raw_folder, exist_ok=True)
        
        if not os.path.exists(self.ref_folder):
            print(f"⚠️ Reference folder not found: {self.ref_folder}")
            os.makedirs(self.ref_folder, exist_ok=True)
        
        # Get sorted file lists (support multiple formats)
        self.raw_files = self._get_image_files(self.raw_folder)
        self.ref_files = self._get_image_files(self.ref_folder)
        
        # Ensure we have matching pairs
        self._validate_pairs()
        
        # Split into train/validation
        self._split_dataset()
        
        print(f"\n✅ DataLoader initialized:")
        print(f"   Total pairs: {len(self.raw_files)}")
        print(f"   Train pairs: {len(self.train_indices)}")
        print(f"   Validation pairs: {len(self.val_indices)}")
        print(f"   Image size: {img_size}x{img_size}")
        print(f"   Batch size: {batch_size}")
        print(f"   Augmentation: {'ON' if augment else 'OFF'}")
        if augment:
            print(f"   Aug profile: {self.augmentation_config['profile']}")

    def _resolve_augmentation_config(self, config):
        """Merge profile presets with explicit augmentation overrides."""
        profile_presets = {
            'none': {
                'flip_prob': 0.0,
                'vertical_flip_prob': 0.0,
                'rotate_prob': 0.0,
                'brightness_prob': 0.0,
                'brightness_delta': 0.0,
                'contrast_prob': 0.0,
                'contrast_lower': 1.0,
                'contrast_upper': 1.0,
            },
            'light': {
                'flip_prob': 0.3,
                'vertical_flip_prob': 0.2,
                'rotate_prob': 0.5,
                'brightness_prob': 0.3,
                'brightness_delta': 0.05,
                'contrast_prob': 0.3,
                'contrast_lower': 0.9,
                'contrast_upper': 1.1,
            },
            'standard': {
                'flip_prob': 0.5,
                'vertical_flip_prob': 0.5,
                'rotate_prob': 1.0,
                'brightness_prob': 0.5,
                'brightness_delta': 0.1,
                'contrast_prob': 0.5,
                'contrast_lower': 0.8,
                'contrast_upper': 1.2,
            },
            'strong': {
                'flip_prob': 0.7,
                'vertical_flip_prob': 0.7,
                'rotate_prob': 1.0,
                'brightness_prob': 0.7,
                'brightness_delta': 0.15,
                'contrast_prob': 0.7,
                'contrast_lower': 0.7,
                'contrast_upper': 1.3,
            },
        }

        cfg = dict(config or {})
        profile = str(cfg.pop('profile', 'standard')).lower()
        if profile not in profile_presets:
            print(f"⚠️ Unknown augmentation profile '{profile}', using 'standard'.")
            profile = 'standard'

        resolved = dict(profile_presets[profile])
        resolved.update(cfg)
        resolved['profile'] = profile
        return resolved
    
    def _get_image_files(self, folder):
        """Get all image files in folder"""
        if not os.path.exists(folder):
            return []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG']
        files = []
        seen = set()
        for ext in extensions:
            for file_path in glob.glob(os.path.join(folder, ext)):
                key = os.path.normcase(os.path.abspath(file_path))
                if key in seen:
                    continue
                seen.add(key)
                files.append(file_path)
        return sorted(files)
    
    def _validate_pairs(self):
        """Ensure we have matching raw and reference pairs"""
        if len(self.raw_files) == 0 or len(self.ref_files) == 0:
            print("⚠️ No images found in one or both folders!")
            self.raw_files = []
            self.ref_files = []
            return
        
        # Extract base names without extensions
        raw_bases = [os.path.splitext(os.path.basename(f))[0] for f in self.raw_files]
        ref_bases = [os.path.splitext(os.path.basename(f))[0] for f in self.ref_files]
        
        # Find common pairs
        common = set(raw_bases) & set(ref_bases)
        
        if len(common) == 0:
            print("⚠️ No matching pairs found! Assuming ordered pairs...")
            # Assume files are in corresponding order
            n_pairs = min(len(self.raw_files), len(self.ref_files))
            self.raw_files = self.raw_files[:n_pairs]
            self.ref_files = self.ref_files[:n_pairs]
        else:
            # Filter to only matching pairs
            self.raw_files = [f for f in self.raw_files 
                            if os.path.splitext(os.path.basename(f))[0] in common]
            self.ref_files = [f for f in self.ref_files 
                            if os.path.splitext(os.path.basename(f))[0] in common]
            # Sort to ensure alignment
            self.raw_files.sort()
            self.ref_files.sort()
        
        print(f"📊 Validated {len(self.raw_files)} valid pairs")
    
    def _split_dataset(self):
        """Split data into train and validation sets"""
        n_samples = len(self.raw_files)
        
        if n_samples == 0:
            self.train_indices = []
            self.val_indices = []
            return
        
        indices = list(range(n_samples))
        
        if self.validation_split > 0 and n_samples > 1:
            try:
                self.train_indices, self.val_indices = train_test_split(
                    indices, test_size=self.validation_split, random_state=42
                )
            except Exception:
                # Fallback if train_test_split fails
                split = int(n_samples * (1 - self.validation_split))
                self.train_indices = indices[:split]
                self.val_indices = indices[split:]
        else:
            self.train_indices = indices
            self.val_indices = []
    
    def load_image(self, image_path, is_raw=True):
        """Load and preprocess a single image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_pair(self, idx):
        """Load a raw-reference pair"""
        if idx >= len(self.raw_files) or idx >= len(self.ref_files):
            raise IndexError(f"Index {idx} out of range")
        
        raw_path = self.raw_files[idx]
        ref_path = self.ref_files[idx]
        
        raw_image = self.load_image(raw_path, is_raw=True)
        ref_image = self.load_image(ref_path, is_raw=False)
        
        return raw_image, ref_image
    
    def augment_pair(self, raw, ref):
        """Apply same augmentation to both raw and reference"""
        if not self.augment:
            return raw, ref

        cfg = self.augmentation_config
        
        # Random horizontal flip
        if tf.random.uniform(()) < cfg['flip_prob']:
            raw = tf.image.flip_left_right(raw)
            ref = tf.image.flip_left_right(ref)
        
        # Random vertical flip
        if tf.random.uniform(()) < cfg['vertical_flip_prob']:
            raw = tf.image.flip_up_down(raw)
            ref = tf.image.flip_up_down(ref)
        
        # Random rotation (90, 180, 270 degrees)
        if tf.random.uniform(()) < cfg['rotate_prob']:
            k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)
            raw = tf.image.rot90(raw, k)
            ref = tf.image.rot90(ref, k)
        
        # Random brightness adjustment (only for raw)
        if cfg['brightness_delta'] > 0 and tf.random.uniform(()) < cfg['brightness_prob']:
            raw = tf.image.random_brightness(raw, max_delta=cfg['brightness_delta'])
        
        # Random contrast adjustment (only for raw)
        if tf.random.uniform(()) < cfg['contrast_prob']:
            raw = tf.image.random_contrast(raw, lower=cfg['contrast_lower'], upper=cfg['contrast_upper'])
        
        # Clip values to [0, 1]
        raw = tf.clip_by_value(raw, 0.0, 1.0)
        ref = tf.clip_by_value(ref, 0.0, 1.0)
        
        return raw, ref
    
    def _generator(self, indices):
        """Generator function for tf.data.Dataset"""
        def gen():
            for idx in indices:
                try:
                    raw, ref = self.load_pair(idx)
                    yield raw, ref
                except Exception as e:
                    print(f"⚠️ Error loading pair {idx}: {e}")
                    continue
        return gen
    
    def get_dataset(self, split='train', shuffle=True):
        """
        Get TensorFlow dataset for specified split
        
        Args:
            split: 'train' or 'validation'
            shuffle: Whether to shuffle the dataset
        """
        indices = self.train_indices if split == 'train' else self.val_indices
        
        if len(indices) == 0:
            return None
        
        # Create dataset from generator
        dataset = tf.data.Dataset.from_generator(
            self._generator(indices),
            output_signature=(
                tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32)
            )
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(len(indices), 100))
        
        # Apply augmentation for training
        if split == 'train' and self.augment:
            dataset = dataset.map(self.augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        
        # Repeat so Keras doesn't warn about running out of data
        dataset = dataset.repeat()
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @property
    def train_steps(self):
        """Number of batches per training epoch"""
        return math.ceil(len(self.train_indices) / self.batch_size)

    @property
    def val_steps(self):
        """Number of batches per validation epoch"""
        if len(self.val_indices) == 0:
            return 0
        return math.ceil(len(self.val_indices) / self.batch_size)
    
    def get_sample_batch(self, split='train', n_samples=4):
        """Get a sample batch for visualization"""
        dataset = self.get_dataset(split, shuffle=True)
        if dataset is None:
            return None, None
        
        for batch_raw, batch_ref in dataset.take(1):
            return batch_raw.numpy()[:n_samples], batch_ref.numpy()[:n_samples]
        
        return None, None
    
    def visualize_samples(self, n_samples=4, save_path=None):
        """Visualize sample raw-reference pairs"""
        raw_samples, ref_samples = self.get_sample_batch('train', n_samples)
        
        if raw_samples is None:
            print("No samples to visualize")
            return
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 5*n_samples))
        
        # Handle single sample case
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            # Raw image
            axes[i, 0].imshow(raw_samples[i])
            axes[i, 0].set_title(f'Raw (Input) - Sample {i+1}')
            axes[i, 0].axis('off')
            
            # Reference image
            axes[i, 1].imshow(ref_samples[i])
            axes[i, 1].set_title(f'Reference (Target) - Sample {i+1}')
            axes[i, 1].axis('off')
        
        plt.suptitle('Underwater Image Pairs', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved visualization to: {save_path}")
        
        plt.show()
    
    def get_statistics(self):
        """Get dataset statistics"""
        stats = {
            'total_pairs': len(self.raw_files),
            'train_pairs': len(self.train_indices),
            'val_pairs': len(self.val_indices),
            'image_size': f"{self.img_size}x{self.img_size}",
            'batch_size': self.batch_size,
            'augmentation': self.augment
        }
        
        # Sample a few images to compute mean/std
        if len(self.raw_files) > 0:
            sample_raw, sample_ref = [], []
            for i in range(min(5, len(self.raw_files))):
                try:
                    raw, ref = self.load_pair(i)
                    sample_raw.append(raw)
                    sample_ref.append(ref)
                except Exception:
                    continue
            
            if sample_raw:
                sample_raw = np.stack(sample_raw)
                sample_ref = np.stack(sample_ref)
                
                stats['raw_mean'] = np.mean(sample_raw, axis=(0, 1, 2)).tolist()
                stats['raw_std'] = np.std(sample_raw, axis=(0, 1, 2)).tolist()
                stats['ref_mean'] = np.mean(sample_ref, axis=(0, 1, 2)).tolist()
                stats['ref_std'] = np.std(sample_ref, axis=(0, 1, 2)).tolist()
        
        return stats

# Test the enhanced loader
if __name__ == "__main__":
    print("="*60)
    print("TESTING ENHANCED UNDERWATER DATA LOADER")
    print("="*60)
    
    # Initialize loader
    loader = UnderwaterDataLoader(
        data_path="data",
        img_size=128,
        batch_size=4,
        validation_split=0.2,
        augment=True
    )
    
    # Get statistics
    stats = loader.get_statistics()
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Get datasets
    train_dataset = loader.get_dataset('train')
    val_dataset = loader.get_dataset('validation')
    
    if train_dataset:
        print("\n📁 Training dataset created")
    if val_dataset:
        print("📁 Validation dataset created")
    
    # Test batch
    if train_dataset:
        print("\n🔄 Testing batch loading...")
        for batch_raw, batch_ref in train_dataset.take(1):
            print(f"   Batch raw shape: {batch_raw.shape}")
            print(f"   Batch ref shape: {batch_ref.shape}")
            print(f"   Raw min/max: {batch_raw.numpy().min():.3f}/{batch_raw.numpy().max():.3f}")
            print(f"   Ref min/max: {batch_ref.numpy().min():.3f}/{batch_ref.numpy().max():.3f}")
    
    print("\n✅ Data loader test complete!")