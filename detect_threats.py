#!/usr/bin/env python3
"""Underwater threat detection: enhancement + YOLO detection."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


def resolve_enhancement_model(model_path: str | None) -> str:
    if model_path and Path(model_path).exists():
        return model_path

    candidates = sorted(Path("models/checkpoints").glob("*_final.keras"), reverse=True)
    if candidates:
        return str(candidates[0])

    candidates = sorted(Path("models/checkpoints").glob("*_final.h5"), reverse=True)
    if candidates:
        return str(candidates[0])

    raise FileNotFoundError("No enhancement model found. Pass --enhance-model explicitly.")


def resolve_yolo_model(model_path: str | None) -> str:
    if model_path and Path(model_path).exists():
        return model_path

    candidates = sorted(
        Path("runs").glob("**/underwater_threat_detector/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])

    return "yolov8n.pt"


class UnderwaterThreatDetector:
    """Complete pipeline: enhancement + YOLO detection."""

    def __init__(
        self,
        enhancement_model_path: str | None = None,
        yolo_model_path: str = "runs/train/underwater_threat_detector/weights/best.pt",
        confidence_threshold: float = 0.5,
        enhance_size: int = 0,
    ):
        print("=" * 60)
        print("UNDERWATER THREAT DETECTION SYSTEM")
        print("=" * 60)

        resolved = resolve_enhancement_model(enhancement_model_path)
        print(f"Loading enhancement model: {resolved}")
        self.enhancement_model = tf.keras.models.load_model(resolved, compile=False)

        inferred_size = 128
        try:
            shape = self.enhancement_model.input_shape
            if isinstance(shape, list) and shape:
                shape = shape[0]
            if len(shape) >= 3 and shape[1] is not None:
                inferred_size = int(shape[1])
        except Exception:
            pass

        self.enhance_size = int(enhance_size) if int(enhance_size) > 0 else inferred_size
        print(f"Enhancement input size: {self.enhance_size}")

        resolved_yolo = resolve_yolo_model(yolo_model_path)
        if resolved_yolo == "yolov8n.pt":
            print("Fine-tuned YOLO model not found. Falling back to yolov8n.pt")
        print(f"Loading YOLO model: {resolved_yolo}")
        self.detector = YOLO(resolved_yolo)

        self.confidence_threshold = float(confidence_threshold)
        self.threat_levels = {
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

        Path("results/detections").mkdir(parents=True, exist_ok=True)

    def enhance_image(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        resized = cv2.resize(image_rgb, (self.enhance_size, self.enhance_size), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        pred = self.enhancement_model.predict(x[np.newaxis, ...], verbose=0)[0]

        pred = np.clip(pred, 0.0, 1.0)
        pred = (pred * 255.0).astype(np.uint8)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
        return pred

    def detect_threats(self, image_bgr: np.ndarray, return_visualization: bool = True):
        enhanced_rgb = self.enhance_image(image_bgr)

        results = self.detector(enhanced_rgb, conf=self.confidence_threshold, verbose=False)

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.detector.names.get(class_id, str(class_id))
            threat_level = self.threat_levels.get(class_name, "UNKNOWN")

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                    "threat_level": threat_level,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            )

        annotated = self.draw_detections(enhanced_rgb, detections) if return_visualization else None
        return detections, annotated

    def draw_detections(self, image_rgb: np.ndarray, detections: list[dict]) -> np.ndarray:
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        colors = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 64, 255),
            "MEDIUM": (0, 215, 255),
            "LOW": (0, 255, 0),
            "UNKNOWN": (255, 255, 255),
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = colors.get(det["threat_level"], (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.2f} [{det['threat_level']}]"
            cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if det["threat_level"] in {"CRITICAL", "HIGH"}:
                cv2.putText(img, "ALERT", (x1, min(img.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return img

    def process_image(self, image_path: str, output_path: str | None = None):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        t0 = time.time()
        detections, annotated = self.detect_threats(image)
        ms = (time.time() - t0) * 1000.0

        print(f"Processed image in {ms:.1f} ms | detections: {len(detections)}")
        for d in detections:
            print(f"- {d['class']} conf={d['confidence']:.2f} level={d['threat_level']}")

        if output_path and annotated is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, annotated)
            print(f"Saved annotated image: {output_path}")

        self._save_json_report(detections, prefix="image")
        return detections, annotated

    def process_video(self, video_path: str, output_path: str | None = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_times = []
        all_detections = []

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.time()
            detections, annotated = self.detect_threats(frame)
            frame_times.append((time.time() - t0) * 1000.0)

            if detections:
                for d in detections:
                    d["frame"] = idx
                all_detections.extend(detections)

            if writer is not None and annotated is not None:
                writer.write(annotated)

            idx += 1
            if idx % 50 == 0:
                pct = (idx / max(1, total)) * 100
                print(f"Progress: {idx}/{total} ({pct:.1f}%)")

        cap.release()
        if writer is not None:
            writer.release()

        avg = float(np.mean(frame_times)) if frame_times else 0.0
        print(f"Video complete. Frames: {idx}, avg {avg:.1f} ms/frame, detections: {len(all_detections)}")

        self._save_json_report(all_detections, prefix="video")
        return all_detections

    def realtime_detection(self, camera_id: int = 0):
        print("Starting real-time detection. Press q to quit, s to save frame.")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        frame_times = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.time()
            detections, annotated = self.detect_threats(frame)
            ms = (time.time() - t0) * 1000.0
            frame_times.append(ms)
            if len(frame_times) > 30:
                frame_times.pop(0)

            fps = 1000.0 / (float(np.mean(frame_times)) + 1e-6)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Threats: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if any(d["threat_level"] in {"CRITICAL", "HIGH"} for d in detections):
                cv2.putText(annotated, "CRITICAL THREAT DETECTED", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow("Underwater Threat Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = Path("results/detections") / f"threat_{ts}.jpg"
                cv2.imwrite(str(save_path), annotated)
                print(f"Saved frame: {save_path}")

        cap.release()
        cv2.destroyAllWindows()

    def _save_json_report(self, detections: list[dict], prefix: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results/detections") / f"{prefix}_detections_{ts}.json"
        out.write_text(json.dumps(detections, indent=2), encoding="utf-8")
        return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Underwater threat detection")
    parser.add_argument("--mode", required=True, choices=["image", "video", "webcam"], help="Execution mode")
    parser.add_argument("--input", type=str, help="Input file path (image/video)")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--enhance-model", type=str, default=None, help="Enhancement model path")
    parser.add_argument("--yolo-model", type=str, default="runs/train/underwater_threat_detector/weights/best.pt", help="YOLO model path")
    parser.add_argument("--enhance-size", type=int, default=0, help="Enhancement model input size (0=auto)")
    parser.add_argument("--camera", type=int, default=0, help="Camera id for webcam mode")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    detector = UnderwaterThreatDetector(
        enhancement_model_path=args.enhance_model,
        yolo_model_path=args.yolo_model,
        confidence_threshold=args.conf,
        enhance_size=args.enhance_size,
    )

    if args.mode == "image":
        if not args.input:
            raise ValueError("--input is required for image mode")
        output = args.output or args.input.replace(".", "_detected.")
        detector.process_image(args.input, output)
        return

    if args.mode == "video":
        if not args.input:
            raise ValueError("--input is required for video mode")
        output = args.output or args.input.replace(".", "_detected.")
        detector.process_video(args.input, output)
        return

    detector.realtime_detection(camera_id=args.camera)


if __name__ == "__main__":
    main()
