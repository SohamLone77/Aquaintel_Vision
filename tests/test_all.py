"""Repository smoke tests for core modules and scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec

import numpy as np


def _has_module(name: str) -> bool:
    return find_spec(name) is not None


class TestImports(unittest.TestCase):
    @unittest.skipUnless(_has_module("tensorflow"), "tensorflow not installed in current environment")
    def test_tensorflow_import(self):
        import tensorflow as tf
        self.assertTrue(hasattr(tf, "__version__"))

    @unittest.skipUnless(_has_module("ultralytics"), "ultralytics not installed in current environment")
    def test_ultralytics_import(self):
        from ultralytics import YOLO
        self.assertIsNotNone(YOLO)

    @unittest.skipUnless(_has_module("tensorflow"), "tensorflow not installed in current environment")
    def test_project_imports(self):
        from models.basic_unet import build_basic_unet
        from losses.simple_losses import SimpleLosses
        self.assertIsNotNone(build_basic_unet)
        self.assertIsNotNone(SimpleLosses)


class TestModel(unittest.TestCase):
    @unittest.skipUnless(_has_module("tensorflow"), "tensorflow not installed in current environment")
    def test_unet_build(self):
        from models.basic_unet import build_basic_unet
        m = build_basic_unet(input_shape=(128, 128, 3))
        self.assertGreater(m.count_params(), 1000)

    @unittest.skipUnless(_has_module("tensorflow"), "tensorflow not installed in current environment")
    def test_unet_infer_shape(self):
        from models.basic_unet import build_basic_unet
        m = build_basic_unet(input_shape=(128, 128, 3))
        x = np.random.rand(1, 128, 128, 3).astype(np.float32)
        y = m.predict(x, verbose=0)
        self.assertEqual(y.shape, (1, 128, 128, 3))


class TestScripts(unittest.TestCase):
    def _run_help(self, script):
        # Fix N8: Guard test subprocess calls by ensuring the script exists
        # and checking that the script parameter is a direct filename.
        if os.path.basename(script) != script:
            raise ValueError("Script must be a simple filename in the root directory")
        if not os.path.exists(script):
            self.skipTest(f"{script} not found")
        
        cmd = [sys.executable, script, "--help"]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_prepare_dataset_help(self):
        r = self._run_help("prepare_dataset.py")
        self.assertEqual(r.returncode, 0)

    def test_train_yolo_help(self):
        if not _has_module("ultralytics"):
            self.skipTest("ultralytics not installed in current environment")
        r = self._run_help("train_yolo.py")
        self.assertEqual(r.returncode, 0)

    def test_detect_threats_help(self):
        if not _has_module("ultralytics"):
            self.skipTest("ultralytics not installed in current environment")
        r = self._run_help("detect_threats.py")
        self.assertEqual(r.returncode, 0)


class TestFilesystem(unittest.TestCase):
    def test_required_paths_exist_or_creatable(self):
        required_dirs = ["models", "losses", "training", "results", "logs"]
        for d in required_dirs:
            os.makedirs(d, exist_ok=True)
            self.assertTrue(os.path.isdir(d))


class TestConfig(unittest.TestCase):
    def test_load_config_valid(self):
        from utils.config_loader import load_runtime_config
        # It should load the project default config.yaml without error
        cfg = load_runtime_config()
        self.assertIn("img_size", cfg)
        self.assertIn("learning_rate", cfg)
        self.assertIn("batch_size", cfg)


class TestRegistry(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile
        from utils.model_registry import ModelRegistry
        with tempfile.TemporaryDirectory() as tmp:
            rp = os.path.join(tmp, "reg.json")
            reg = ModelRegistry(rp)
            reg.register_training_run("dummy_run", {"epochs": 1}, {"loss": 0.5}, {})
            
            # Re-read
            reg2 = ModelRegistry(rp)
            self.assertIn("dummy_run", reg2.records)
            self.assertEqual(reg2.records["dummy_run"]["metrics"]["loss"], 0.5)


class TestDataPipeline(unittest.TestCase):
    @unittest.skipUnless(_has_module("tensorflow"), "Missing TF")
    def test_overlapping_indices(self):
        from training.data_loader import UnderwaterDataLoader
        # Assuming there is a data folder or we just test the split
        if not os.path.exists("data"):
            self.skipTest("No data folder found to test dataloader")
            
        loader = UnderwaterDataLoader(data_path="data", img_size=128, batch_size=2, validation_split=0.2)
        # Check that there is no overlap in train/val
        overlap = set(loader.train_indices) & set(loader.val_indices)
        self.assertEqual(len(overlap), 0, "Train and Val indices overlap!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
