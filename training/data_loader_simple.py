"""Simple high-resolution data loader for paired underwater enhancement datasets."""

import glob
import os

import cv2
import numpy as np
import tensorflow as tf


class SimpleDataLoader:
    """Data loader that supports configurable resolution and batch size."""

    def __init__(self, data_path="data", img_size=256, batch_size=2, validation_split=0.2):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split

        self.raw_folder = os.path.join(data_path, "raw")
        self.ref_folder = os.path.join(data_path, "reference")

        if not os.path.exists(self.raw_folder):
            print(f"Raw folder not found: {self.raw_folder}")
            os.makedirs(self.raw_folder, exist_ok=True)
            print(f"Created: {self.raw_folder}")

        if not os.path.exists(self.ref_folder):
            print(f"Reference folder not found: {self.ref_folder}")
            os.makedirs(self.ref_folder, exist_ok=True)
            print(f"Created: {self.ref_folder}")

        self.raw_files = self._get_image_files(self.raw_folder)
        self.ref_files = self._get_image_files(self.ref_folder)

        n_pairs = min(len(self.raw_files), len(self.ref_files))
        self.raw_files = self.raw_files[:n_pairs]
        self.ref_files = self.ref_files[:n_pairs]

        print(f"Found {len(self.raw_files)} image pairs")
        print(f"Image size: {img_size}x{img_size}")
        print(f"Batch size: {batch_size}")

        self._split_indices()

    def _get_image_files(self, folder):
        if not os.path.exists(folder):
            return []

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(files)

    def _split_indices(self):
        n_samples = len(self.raw_files)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        split = int(n_samples * (1 - self.validation_split))
        self.train_indices = indices[:split]
        self.val_indices = indices[split:]

        print(f"Training samples: {len(self.train_indices)}")
        print(f"Validation samples: {len(self.val_indices)}")

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            print(f"Could not load: {path}")
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        return img.astype(np.float32) / 255.0

    def _generator(self, indices):
        def gen():
            for idx in indices:
                raw_path = self.raw_files[idx]
                ref_path = self.ref_files[idx]

                raw_img = self.load_image(raw_path)
                ref_img = self.load_image(ref_path)
                yield raw_img, ref_img

        return gen

    def get_dataset(self, split="train"):
        indices = self.train_indices if split == "train" else self.val_indices

        if len(indices) == 0:
            print(f"No {split} data available")
            return None

        dataset = tf.data.Dataset.from_generator(
            self._generator(indices),
            output_signature=(
                tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.img_size, self.img_size, 3), dtype=tf.float32),
            ),
        )

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Data Loader")
    print("=" * 50)

    loader = SimpleDataLoader(data_path="data", img_size=256, batch_size=2, validation_split=0.2)
    train_ds = loader.get_dataset("train")

    if train_ds is not None:
        for x, y in train_ds.take(1):
            print(f"Batch shapes: {x.shape}, {y.shape}")
            print(f"Value range: [{x.numpy().min():.3f}, {x.numpy().max():.3f}]")

    print("\nData loader test complete")