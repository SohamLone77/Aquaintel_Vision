#!/usr/bin/env python3
"""
Production YOLO detection pipeline for underwater threat detection.

Recommended deployment settings from in-repo validation:
- Enhancement OFF
- Confidence threshold 0.30
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from utils.config_loader import load_runtime_config

try:
    _cfg = load_runtime_config()
    _cfg_yolo_path = _cfg.get("yolo_model_path", "runs/detect/runs/yolo/underwater_run5_20260413/weights/best.pt")
except Exception:
    _cfg_yolo_path = "runs/detect/runs/yolo/underwater_run5_20260413/weights/best.pt"

DEFAULT_YOLO_PATH = os.environ.get("YOLO_MODEL_PATH", _cfg_yolo_path)
DEFAULT_REPORT_PATH = "results/detection_report.json"

DEFAULT_CLASS_THRESHOLDS = {
    "diver": 0.30,
    "suspicious_object": 0.25,
    "boat": 0.35,
    "ship": 0.35,
    "submarine": 0.25,
    "pipeline": 0.40,
    "obstacle": 0.35,
    "marine_life": 0.40,
    "swimmer": 0.25,
    "underwater_vehicle": 0.25,
}

THREAT_LEVELS = {
    "diver": "HIGH",
    "suspicious_object": "CRITICAL",
    "boat": "MEDIUM",
    "ship": "LOW",
    "submarine": "CRITICAL",
    "pipeline": "LOW",
    "obstacle": "MEDIUM",
    "marine_life": "LOW",
    "swimmer": "HIGH",
    "underwater_vehicle": "MEDIUM",
}

PRETRAINED_THREAT_MAPPING = {
    0: {"name": "diver", "threat": "HIGH"},
    9: {"name": "boat", "threat": "MEDIUM"},
    10: {"name": "ship", "threat": "LOW"},
}

THREAT_COLORS = {
    "CRITICAL": (0, 0, 255),
    "HIGH": (0, 64, 255),
    "MEDIUM": (0, 215, 255),
    "LOW": (0, 255, 0),
    "UNKNOWN": (255, 255, 255),
}


def resolve_yolo_model(model_path: str | None, allow_fallback: bool = True) -> str:
    if model_path:
        if Path(model_path).exists():
            return model_path
        if not allow_fallback:
            raise FileNotFoundError(f"YOLO weights not found: {model_path}")

    preferred = sorted(
        Path("runs").glob("**/underwater_run5_20260413/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return str(preferred[0])

    candidates = sorted(
        Path("runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])

    if not allow_fallback:
        raise FileNotFoundError("No YOLO weights found. Pass --model explicitly.")

    raise FileNotFoundError("No YOLO weights found. Pass --model explicitly.")


def resolve_enhancement_model(model_path: str | None) -> str:
    if model_path and Path(model_path).exists():
        return model_path

    candidates = sorted(Path("models/checkpoints").glob("*_final.keras"), reverse=True)
    if candidates:
        return str(candidates[0])

    candidates = sorted(Path("models/checkpoints").glob("*_final.h5"), reverse=True)
    if candidates:
        return str(candidates[0])

    raise FileNotFoundError(
        "No enhancement model found. Pass --enhance-model explicitly or disable enhancement."
    )


def load_yaml_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for --config. Install pyyaml.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def cfg_get(cfg: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


class UnderwaterThreatDetector:
    """Production-ready threat detection system."""

    def __init__(
        self,
        yolo_model_path: str | None = None,
        enhancement_model_path: str | None = None,
        confidence_threshold: float = 0.30,
        stage1_confidence: float | None = None,
        stage2_confidence: float | None = None,
        temporal_frames: int = 1,
        min_consistent_frames: int = 1,
        alert_cooldown: float = 0.0,
        iou_threshold: float = 0.45,
        enable_enhancement: bool = False,
        class_thresholds: dict[str, float] | None = None,
        stage2_thresholds: dict[str, float] | None = None,
        use_pretrained: bool = False,
        pretrained_threat_mapping: dict[int, dict[str, str]] | None = None,
    ) -> None:
        print("=" * 66)
        print("UNDERWATER THREAT DETECTION SYSTEM")
        print("=" * 66)

        self.use_pretrained = bool(use_pretrained or pretrained_threat_mapping)
        if pretrained_threat_mapping:
            self.pretrained_threat_mapping = {
                int(k): dict(v) for k, v in pretrained_threat_mapping.items()
            }
        elif self.use_pretrained:
            self.pretrained_threat_mapping = {
                k: dict(v) for k, v in PRETRAINED_THREAT_MAPPING.items()
            }
        else:
            self.pretrained_threat_mapping = {}

        resolved_yolo = resolve_yolo_model(
            yolo_model_path or DEFAULT_YOLO_PATH,
            allow_fallback=not self.use_pretrained,
        )
        print(f"Loading YOLO model: {resolved_yolo}")
        self.detector = YOLO(resolved_yolo)
        self.model_path = resolved_yolo

        self.enable_enhancement = bool(enable_enhancement)
        self.enhancement_model = None
        self.enhance_size = 0
        self._tf = None

        if self.enable_enhancement:
            resolved_enh = resolve_enhancement_model(enhancement_model_path)
            print(f"Loading enhancement model: {resolved_enh}")
            try:
                import tensorflow as tf  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "TensorFlow is required when enhancement is enabled."
                ) from exc

            self._tf = tf
            self.enhancement_model = tf.keras.models.load_model(resolved_enh, compile=False)

            inferred_size = 128
            try:
                shape = self.enhancement_model.input_shape
                if isinstance(shape, list) and shape:
                    shape = shape[0]
                if len(shape) >= 3 and shape[1] is not None:
                    inferred_size = int(shape[1])
            except Exception:
                pass

            self.enhance_size = inferred_size
            print(f"Enhancement input size: {self.enhance_size}")
        else:
            print("Enhancement: OFF (recommended based on evaluation)")

        self.confidence_threshold = float(confidence_threshold)
        self.stage1_confidence = float(stage1_confidence) if stage1_confidence is not None else self.confidence_threshold
        self.stage2_confidence = float(stage2_confidence) if stage2_confidence is not None else self.confidence_threshold
        self.iou_threshold = float(iou_threshold)
        self.class_thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
        if class_thresholds:
            self.class_thresholds.update(class_thresholds)
        self.stage2_thresholds = dict(stage2_thresholds or {})

        self.temporal_frames = max(1, int(temporal_frames))
        self.min_consistent_frames = max(1, int(min_consistent_frames))
        self.alert_cooldown = max(0.0, float(alert_cooldown))
        self.detection_history = deque(maxlen=self.temporal_frames)
        self.last_alert_time: dict[str, float] = {}

        self.threat_levels = dict(THREAT_LEVELS)
        self.colors = dict(THREAT_COLORS)

        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "stage1_detections": 0,
            "stage2_alerts": 0,
            "class_counts": defaultdict(int),
            "processing_times": [],
            "alerts_triggered": [],
        }

        Path("results/detections").mkdir(parents=True, exist_ok=True)
        Path("logs/detection").mkdir(parents=True, exist_ok=True)

        print("Configuration:")
        print(f"  enhancement: {'ON' if self.enable_enhancement else 'OFF'}")
        print(f"  pretrained:  {'ON' if self.pretrained_threat_mapping else 'OFF'}")
        print(f"  stage1 conf: {self.stage1_confidence}")
        print(f"  stage2 conf: {self.stage2_confidence}")
        print(f"  iou:         {self.iou_threshold}")

    def enhance_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.enable_enhancement or self.enhancement_model is None:
            return image_bgr

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        resized = cv2.resize(image_rgb, (self.enhance_size, self.enhance_size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        pred = self.enhancement_model.predict(x[np.newaxis, ...], verbose=0)[0]

        pred = np.clip(pred, 0.0, 1.0)
        pred = (pred * 255.0).astype(np.uint8)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

    def detect(
        self,
        image_bgr: np.ndarray,
        apply_enhancement: bool | None = None,
        use_temporal: bool = False,
    ) -> tuple[list[dict[str, Any]], np.ndarray, float]:
        start = time.time()

        use_enhancement = self.enable_enhancement if apply_enhancement is None else bool(apply_enhancement)
        if use_enhancement and self.enhancement_model is not None:
            processed = self.enhance_image(image_bgr)
        else:
            processed = image_bgr

        result = self.detector(
            processed,
            conf=self.stage1_confidence,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        stage1_detections: list[dict[str, Any]] = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.detector.names.get(class_id, str(class_id))
            threat_level = self.threat_levels.get(class_name, "UNKNOWN")

            if self.pretrained_threat_mapping:
                mapping = self.pretrained_threat_mapping.get(class_id)
                if not mapping:
                    continue
                class_name = mapping.get("name", class_name)
                threat_level = mapping.get("threat", threat_level)

            per_class_threshold = float(self.class_thresholds.get(class_name, self.stage1_confidence))
            if confidence < per_class_threshold:
                continue
            stage1_detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "threat_level": threat_level,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )
            self.stats["class_counts"][class_name] += 1

        stage2_detections = []
        for det in stage1_detections:
            class_name = det["class_name"]
            min_conf = float(self.stage2_thresholds.get(class_name, self.stage2_confidence))
            if det["confidence"] >= min_conf:
                stage2_detections.append(det)

        self.stats["stage1_detections"] += len(stage1_detections)
        self.stats["stage2_alerts"] += len(stage2_detections)

        final_detections = stage2_detections
        if use_temporal and (self.temporal_frames > 1 or self.min_consistent_frames > 1):
            final_detections = self._temporal_smooth(stage2_detections)
            final_detections = self._apply_alert_cooldown(final_detections)

        for alert in final_detections:
            self.stats["alerts_triggered"].append(
                {
                    "timestamp": time.time(),
                    "class": alert["class_name"],
                    "confidence": alert["confidence"],
                    "threat_level": alert["threat_level"],
                }
            )

        annotated = self.draw_detections(processed, final_detections)
        elapsed_ms = (time.time() - start) * 1000.0

        self.stats["total_frames"] += 1
        self.stats["total_detections"] += len(final_detections)
        self.stats["processing_times"].append(elapsed_ms)

        return final_detections, annotated, elapsed_ms

    def _temporal_smooth(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.temporal_frames <= 1 or self.min_consistent_frames <= 1:
            return detections

        self.detection_history.append(detections)
        if len(self.detection_history) < self.min_consistent_frames:
            return []

        detection_counts: dict[tuple[str, tuple[int, int, int, int]], int] = {}
        for frame_dets in self.detection_history:
            for det in frame_dets:
                key = (det["class_name"], tuple(det["bbox"]))
                detection_counts[key] = detection_counts.get(key, 0) + 1

        stable: list[dict[str, Any]] = []
        for det in detections:
            key = (det["class_name"], tuple(det["bbox"]))
            if detection_counts.get(key, 0) >= self.min_consistent_frames:
                stable.append(det)

        return stable

    def _apply_alert_cooldown(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.alert_cooldown <= 0:
            return detections

        filtered: list[dict[str, Any]] = []
        now = time.time()
        for det in detections:
            x1, y1, _, _ = det["bbox"]
            key = f"{det['class_name']}_{x1}_{y1}"
            last = self.last_alert_time.get(key, 0.0)
            if now - last < self.alert_cooldown:
                continue
            self.last_alert_time[key] = now
            filtered.append(det)

        return filtered

    def draw_detections(self, image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        out = image_bgr.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = self.colors.get(det["threat_level"], self.colors["UNKNOWN"])

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f} [{det['threat_level']}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            top = max(0, y1 - lh - 8)
            cv2.rectangle(out, (x1, top), (x1 + lw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        mode = "ENH:ON" if self.enable_enhancement else "ENH:OFF"
        cv2.putText(
            out,
            f"{mode} S1:{self.stage1_confidence:.2f} S2:{self.stage2_confidence:.2f} IOU:{self.iou_threshold:.2f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        return out

    def process_image(
        self,
        image_path: str,
        output_path: str | None = None,
        visualize: bool = False,
    ) -> list[dict[str, Any]]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        detections, annotated, elapsed_ms = self.detect(image, use_temporal=False)
        print(f"Processed image in {elapsed_ms:.1f} ms | detections: {len(detections)}")
        for det in detections:
            print(f"- {det['class_name']} conf={det['confidence']:.2f} level={det['threat_level']}")

        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), annotated)
            print(f"Saved: {out_path}")

        if visualize:
            cv2.imshow("Threat Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections

    def process_video(
        self,
        video_path: str,
        output_path: str | None = None,
        skip_frames: int = 1,
        max_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        frame_idx = 0
        processed = 0
        all_detections: list[dict[str, Any]] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % max(1, int(skip_frames)) == 0:
                detections, annotated, _ = self.detect(frame, use_temporal=True)
                all_detections.extend(detections)
                processed += 1
                if writer is not None:
                    writer.write(annotated)

                if processed % 50 == 0 and total_frames > 0:
                    pct = (frame_idx / total_frames) * 100.0
                    print(f"Progress: {pct:.1f}% ({processed} frames processed)")

            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

        cap.release()
        if writer is not None:
            writer.release()

        avg_ms = float(np.mean(self.stats["processing_times"])) if self.stats["processing_times"] else 0.0
        print("Video processing complete")
        print(f"  frames processed: {processed}")
        print(f"  total detections: {len(all_detections)}")
        print(f"  avg inference:    {avg_ms:.1f} ms/frame")

        return all_detections

    def process_webcam(self, camera_id: int = 0) -> None:
        print("Real-time detection started")
        print("Controls: q=quit, s=save frame, e=toggle enhancement")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        frame_times: list[float] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.time()
            detections, annotated, _ = self.detect(frame, use_temporal=True)
            ms = (time.time() - t0) * 1000.0
            frame_times.append(ms)
            if len(frame_times) > 30:
                frame_times.pop(0)

            fps = 1000.0 / (float(np.mean(frame_times)) + 1e-6)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Threats: {len(detections)}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if any(d["threat_level"] in {"CRITICAL", "HIGH"} for d in detections):
                cv2.putText(annotated, "THREAT DETECTED", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Underwater Threat Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path("results/detections") / f"threat_{ts}.jpg"
                cv2.imwrite(str(save_path), annotated)
                print(f"Saved frame: {save_path}")
            if key == ord("e"):
                if self.enhancement_model is None:
                    print("Enhancement model not loaded; cannot enable enhancement")
                else:
                    self.enable_enhancement = not self.enable_enhancement
                    print(f"Enhancement: {'ON' if self.enable_enhancement else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()

    def compare_modes(
        self,
        image_path: str,
        output_path: str | None = None,
        visualize: bool = False,
    ) -> dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        det_off, ann_off, ms_off = self.detect(image, apply_enhancement=False, use_temporal=False)

        if self.enhancement_model is None:
            print("Enhancement model not loaded; compare will include OFF mode only")
            summary = {
                "off": {"count": len(det_off), "ms": ms_off, "detections": det_off},
                "on": None,
            }
            if output_path:
                out_path = Path(output_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), ann_off)
            if visualize:
                cv2.imshow("Compare (OFF only)", ann_off)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return summary

        det_on, ann_on, ms_on = self.detect(image, apply_enhancement=True, use_temporal=False)
        comparison = np.hstack([ann_off, ann_on])
        cv2.putText(comparison, "ENH OFF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(
            comparison,
            "ENH ON",
            (comparison.shape[1] // 2 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), comparison)
            print(f"Saved compare image: {out_path}")

        if visualize:
            cv2.imshow("Compare OFF (left) vs ON (right)", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "off": {"count": len(det_off), "ms": ms_off, "detections": det_off},
            "on": {"count": len(det_on), "ms": ms_on, "detections": det_on},
        }

    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        limit: int = 50,
        visualize: bool = False,
    ) -> list[dict[str, Any]]:
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
            images.extend(sorted(in_dir.glob(pattern)))

        results: list[dict[str, Any]] = []
        for image_path in images[: max(1, int(limit))]:
            output_path = out_dir / f"detected_{image_path.name}"
            detections = self.process_image(str(image_path), str(output_path), visualize=visualize)
            results.append(
                {
                    "image": image_path.name,
                    "detections": len(detections),
                    "classes": [d["class_name"] for d in detections],
                }
            )
        return results

    def generate_report(self, output_path: str = DEFAULT_REPORT_PATH) -> dict[str, Any]:
        avg_ms = float(np.mean(self.stats["processing_times"])) if self.stats["processing_times"] else 0.0

        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "configuration": {
                "model_path": self.model_path,
                "enable_enhancement": self.enable_enhancement,
                "use_pretrained": bool(self.pretrained_threat_mapping),
                "pretrained_mapping": self.pretrained_threat_mapping or None,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "class_thresholds": self.class_thresholds,
            },
            "statistics": {
                "total_frames": self.stats["total_frames"],
                "total_detections": self.stats["total_detections"],
                "stage1_detections": self.stats["stage1_detections"],
                "stage2_alerts": self.stats["stage2_alerts"],
                "avg_processing_time_ms": avg_ms,
                "detections_per_frame": (
                    self.stats["total_detections"] / max(1, self.stats["total_frames"])
                ),
                "class_counts": dict(self.stats["class_counts"]),
            },
        }

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved: {out_path}")
        return report


def output_path_for_input(input_path: str, suffix: str = "_detected") -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


def build_detector_from_args(args: argparse.Namespace, cfg: dict[str, Any]) -> UnderwaterThreatDetector:
    use_pretrained = bool(args.pretrained)
    if not args.pretrained and "use_pretrained" in (cfg_get(cfg, ["detection"], {}) or {}):
        use_pretrained = bool(cfg_get(cfg, ["detection", "use_pretrained"], False))

    model_path = args.model or cfg_get(cfg, ["model", "yolo_path"], None)
    if use_pretrained and not model_path:
        model_path = "yolo11s.pt"
    if not model_path:
        model_path = DEFAULT_YOLO_PATH
    enhancement_model_path = args.enhance_model or cfg_get(cfg, ["model", "enhancement_path"], None)
    confidence = float(args.conf if args.conf is not None else cfg_get(cfg, ["detection", "confidence"], 0.30))
    stage1_confidence = float(
        args.stage1_conf
        if args.stage1_conf is not None
        else cfg_get(cfg, ["detection", "stage1_confidence"], confidence)
    )
    stage2_confidence = float(
        args.stage2_conf
        if args.stage2_conf is not None
        else cfg_get(cfg, ["detection", "stage2_confidence"], confidence)
    )
    iou = float(args.iou if args.iou is not None else cfg_get(cfg, ["detection", "iou"], 0.45))
    class_thresholds = cfg_get(cfg, ["class_thresholds"], None)
    stage2_thresholds = cfg_get(cfg, ["stage2_thresholds"], None)
    temporal_frames = int(
        args.temporal_frames
        if args.temporal_frames is not None
        else cfg_get(cfg, ["detection", "temporal_frames"], 1)
    )
    min_consistent_frames = int(
        args.min_consistent_frames
        if args.min_consistent_frames is not None
        else cfg_get(cfg, ["detection", "min_consistent_frames"], 1)
    )
    alert_cooldown = float(
        args.alert_cooldown
        if args.alert_cooldown is not None
        else cfg_get(cfg, ["detection", "alert_cooldown"], 0)
    )

    enable_enhancement = bool(args.enhance)
    if not args.enhance and "enable_enhancement" in (cfg_get(cfg, ["detection"], {}) or {}):
        enable_enhancement = bool(cfg_get(cfg, ["detection", "enable_enhancement"], False))

    return UnderwaterThreatDetector(
        yolo_model_path=model_path,
        enhancement_model_path=enhancement_model_path,
        confidence_threshold=confidence,
        stage1_confidence=stage1_confidence,
        stage2_confidence=stage2_confidence,
        temporal_frames=temporal_frames,
        min_consistent_frames=min_consistent_frames,
        alert_cooldown=alert_cooldown,
        iou_threshold=iou,
        enable_enhancement=enable_enhancement,
        class_thresholds=class_thresholds,
        stage2_thresholds=stage2_thresholds,
        use_pretrained=use_pretrained,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Underwater Threat Detection System")
    parser.add_argument("--mode", required=True, choices=["image", "video", "webcam", "compare", "batch"], help="Run mode")
    parser.add_argument("--input", type=str, help="Input image/video/folder path")
    parser.add_argument("--output", type=str, help="Output path")

    parser.add_argument("--model", type=str, default=None, help="YOLO model path")
    parser.add_argument("--enhance-model", type=str, default=None, help="Enhancement model path")
    parser.add_argument("--enhance", action="store_true", help="Enable enhancement")
    parser.add_argument("--pretrained", action="store_true", help="Use COCO pretrained mapping for threats")

    parser.add_argument("--conf", type=float, default=None, help="Global confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--stage1-conf", type=float, default=None, help="Stage-1 confidence threshold")
    parser.add_argument("--stage2-conf", type=float, default=None, help="Stage-2 confidence threshold")
    parser.add_argument("--temporal-frames", type=int, default=None, help="Temporal history length")
    parser.add_argument("--min-consistent-frames", type=int, default=None, help="Min frames for stable detection")
    parser.add_argument("--alert-cooldown", type=float, default=None, help="Cooldown seconds per alert")

    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame (video)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (video)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam id")
    parser.add_argument("--batch-limit", type=int, default=50, help="Max images in batch mode")
    parser.add_argument("--no-visualize", action="store_true", help="Disable visualization windows")

    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT_PATH, help="Report output path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    detector = build_detector_from_args(args, cfg)

    visualize = not args.no_visualize

    if args.mode == "image":
        if not args.input:
            raise ValueError("--input is required for image mode")
        out = args.output or output_path_for_input(args.input)
        detector.process_image(args.input, out, visualize=visualize)
        detector.generate_report(args.report)
        return

    if args.mode == "video":
        if not args.input:
            raise ValueError("--input is required for video mode")
        out = args.output or output_path_for_input(args.input)
        detector.process_video(args.input, out, skip_frames=args.skip_frames, max_frames=args.max_frames)
        detector.generate_report(args.report)
        return

    if args.mode == "webcam":
        detector.process_webcam(camera_id=args.camera)
        detector.generate_report(args.report)
        return

    if args.mode == "compare":
        if not args.input:
            raise ValueError("--input is required for compare mode")
        out = args.output or output_path_for_input(args.input, suffix="_compare")
        detector.compare_modes(args.input, out, visualize=visualize)
        detector.generate_report(args.report)
        return

    if args.mode == "batch":
        input_dir = args.input or "data/sample/raw"
        output_dir = args.output or "results/detections/batch"
        detector.batch_process(input_dir, output_dir, limit=args.batch_limit, visualize=visualize)
        detector.generate_report(args.report)
        return


if __name__ == "__main__":
    main()
