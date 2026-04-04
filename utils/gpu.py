"""TensorFlow device configuration helpers."""

from __future__ import annotations

import os
from typing import Dict

import tensorflow as tf


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def configure_tensorflow_device(config: dict | None = None) -> Dict[str, object]:
    """Configure TensorFlow device strategy and return runtime device details."""
    config = config or {}

    prefer_gpu = bool(config.get("gpu_enabled", _env_flag("USE_GPU", True)))
    memory_growth = bool(config.get("gpu_memory_growth", _env_flag("GPU_MEMORY_GROWTH", True)))
    mixed_precision_enabled = bool(
        config.get("mixed_precision", _env_flag("MIXED_PRECISION", False))
    )

    if not prefer_gpu:
        tf.config.set_visible_devices([], "GPU")
        return {
            "device": "CPU",
            "gpu_count": 0,
            "mixed_precision": False,
        }

    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        return {
            "device": "CPU",
            "gpu_count": 0,
            "mixed_precision": False,
        }

    if memory_growth:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Safe fallback if memory growth was already configured.
                pass

    if mixed_precision_enabled:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")

    logical_gpus = tf.config.list_logical_devices("GPU")
    return {
        "device": "GPU",
        "gpu_count": len(logical_gpus),
        "mixed_precision": mixed_precision_enabled,
    }