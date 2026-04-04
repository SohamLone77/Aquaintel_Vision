"""Utilities for storing training run metadata."""

import json
import os
from datetime import datetime


class ModelRegistry:
    """Simple JSON-backed training metadata registry."""

    def __init__(self, registry_path="results/model_registry.json"):
        self.registry_path = registry_path
        self.records = self._load()

    def _load(self):
        if not os.path.exists(self.registry_path):
            return {}
        try:
            with open(self.registry_path, "r", encoding="utf-8") as file_obj:
                data = json.load(file_obj)
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def register_training_run(self, run_name, config, metrics, artifacts):
        self.records[run_name] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": config,
            "metrics": metrics,
            "artifacts": artifacts,
        }
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as file_obj:
            json.dump(self.records, file_obj, indent=2)
