"""Configuration loading and validation helpers for training scripts."""

import os
from datetime import datetime
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Raised when runtime configuration is missing or invalid."""


_REQUIRED_SCHEMA = {
    "data": {"path", "img_size", "validation_split"},
    "training": {
        "batch_size",
        "epochs",
        "learning_rate",
        "early_stopping_patience",
        "reduce_lr_patience",
    },
    "loss": {"type", "ssim_weight"},
    "augmentation": {
        "enabled",
        "profile",
        "flip_prob",
        "vertical_flip_prob",
        "rotate_prob",
        "brightness_prob",
        "brightness_delta",
        "contrast_prob",
        "contrast_lower",
        "contrast_upper",
    },
    "model": {"name_prefix", "checkpoint_dir", "results_dir", "registry_path"},
    "logging": {"use_tensorboard", "use_csv_logger"},
}


def _validate_schema(config):
    """Validate required sections and keys in the yaml config."""
    missing_sections = [section for section in _REQUIRED_SCHEMA if section not in config]
    if missing_sections:
        sections = ", ".join(missing_sections)
        raise ConfigError(f"Missing required config section(s): {sections}")

    for section, keys in _REQUIRED_SCHEMA.items():
        section_values = config.get(section, {})
        missing_keys = [key for key in keys if key not in section_values]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ConfigError(
                f"Missing required key(s) in '{section}': {missing}"
            )


def _read_yaml(config_path):
    if not Path(config_path).exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as file_obj:
            data = yaml.safe_load(file_obj)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {config_path}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read config file: {config_path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("Config must be a YAML mapping with top-level sections")

    return data


def _flatten_config(config):
    """Convert nested config schema into trainer-compatible flat keys."""
    data_cfg = config["data"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    aug_cfg = config["augmentation"]
    model_cfg = config["model"]
    logging_cfg = config["logging"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return {
        "data_path": os.environ.get("DATA_PATH", data_cfg["path"]),
        "img_size": data_cfg["img_size"],
        "validation_split": data_cfg["validation_split"],
        "batch_size": train_cfg["batch_size"],
        "epochs": train_cfg["epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "early_stopping_patience": train_cfg["early_stopping_patience"],
        "reduce_lr_patience": train_cfg["reduce_lr_patience"],
        "loss_type": loss_cfg["type"],
        "ssim_weight": loss_cfg["ssim_weight"],
        "augment_enabled": aug_cfg["enabled"],
        "augmentation_profile": aug_cfg["profile"],
        "augmentation": {
            "flip_prob": aug_cfg["flip_prob"],
            "vertical_flip_prob": aug_cfg["vertical_flip_prob"],
            "rotate_prob": aug_cfg["rotate_prob"],
            "brightness_prob": aug_cfg["brightness_prob"],
            "brightness_delta": aug_cfg["brightness_delta"],
            "contrast_prob": aug_cfg["contrast_prob"],
            "contrast_lower": aug_cfg["contrast_lower"],
            "contrast_upper": aug_cfg["contrast_upper"],
        },
        "model_name": f"{model_cfg['name_prefix']}_{timestamp}",
        "checkpoint_dir": model_cfg["checkpoint_dir"],
        "results_dir": model_cfg["results_dir"],
        "registry_path": model_cfg["registry_path"],
        "use_tensorboard": logging_cfg["use_tensorboard"],
        "use_csv_logger": logging_cfg["use_csv_logger"],
    }


def load_runtime_config(config_path="config.yaml", overrides=None):
    """Load config from yaml, validate it, and return trainer-ready settings."""
    raw_config = _read_yaml(config_path)
    _validate_schema(raw_config)

    runtime_config = _flatten_config(raw_config)
    if overrides:
        runtime_config.update(overrides)

    return runtime_config
