"""Utilities for storing training run metadata."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


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
        out_path = Path(self.registry_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file first, then atomically rename so a
        # mid-write crash cannot leave a partially-written (corrupted) registry.
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=out_path.parent, prefix=".registry_tmp_", suffix=".json"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
            Path(tmp_name).replace(out_path)
        except Exception:
            # Clean up orphaned temp file on failure.
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
            raise
