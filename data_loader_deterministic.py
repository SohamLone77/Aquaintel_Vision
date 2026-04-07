"""Deterministic data loader with optional aspect-ratio preserving preprocessing."""

from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import tensorflow as tf


class DeterministicDataLoader:
    def __init__(
        self,
        data_path="data",
        img_size=256,
        batch_size=2,
        validation_split=0.2,
        deterministic=False,
        seed=42,
        preserve_aspect_ratio=False,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.deterministic = deterministic
        self.preserve_aspect_ratio = preserve_aspect_ratio

        if deterministic:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.raw_folder = os.path.join(data_path, "raw")
        self.ref_folder = os.path.join(data_path, "reference")
        self.raw_files = self._get_image_files(self.raw_folder)
        self.ref_files = self._get_image_files(self.ref_folder)
        n = min(len(self.raw_files), len(self.ref_files))
        self.raw_files = self.raw_files[:n]
        self.ref_files = self.ref_files[:n]
        self._split_indices(seed)

    def _get_image_files(self, folder):
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        return sorted(set(files))

    def _split_indices(self, seed):
        n = len(self.raw_files)
        indices = np.arange(n)
        if not self.deterministic:
            np.random.shuffle(indices)
        split = int(n * (1 - self.validation_split))
        self.train_indices = indices[:split]
        self.val_indices = indices[split:]

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.preserve_aspect_ratio:
            h, w = img.shape[:2]
            scale = self.img_size / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
            canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            y0, x0 = (self.img_size - nh) // 2, (self.img_size - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = resized
            img = canvas
        else:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)

        return img.astype(np.float32) / 255.0

    def _generator(self, indices):
        def gen():
            for i in indices:
                yield self._load_image(self.raw_files[i]), self._load_image(self.ref_files[i])

        return gen

    def get_dataset(self, split="train", shuffle=True, cache=True, prefetch=True):
        indices = self.train_indices if split == "train" else self.val_indices
        if len(indices) == 0:
            return None
        ds = tf.data.Dataset.from_generator(
            self._generator(indices),
            output_types=(tf.float32, tf.float32),
            output_shapes=((self.img_size, self.img_size, 3), (self.img_size, self.img_size, 3)),
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(indices), 1000), seed=42 if self.deterministic else None)
        if cache:
            ds = ds.cache()
        ds = ds.batch(self.batch_size)
        if prefetch:
            ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    @property
    def train_steps(self):
        return max(1, len(self.train_indices) // max(1, self.batch_size))

    @property
    def val_steps(self):
        if len(self.val_indices) == 0:
            return 0
        return max(1, len(self.val_indices) // max(1, self.batch_size))
