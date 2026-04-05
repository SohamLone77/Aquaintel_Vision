"""Compatibility wrapper around canonical threat detector with batched video helper."""

from __future__ import annotations

from collections import Counter, deque
from datetime import datetime

import cv2

from detect_threats import UnderwaterThreatDetector


class EnhancedThreatDetector(UnderwaterThreatDetector):
    def __init__(
        self,
        enhancement_model_path,
        yolo_model_path="yolov8n.pt",
        confidence_threshold=0.5,
        batch_size=4,
        smoothing_window=5,
    ):
        super().__init__(
            enhancement_model_path=enhancement_model_path,
            yolo_model_path=yolo_model_path,
            confidence_threshold=confidence_threshold,
        )
        self.batch_size = int(batch_size)
        self.history = deque(maxlen=max(1, int(smoothing_window)))

    def _smooth(self, detections):
        self.history.append(detections)
        if len(self.history) < self.history.maxlen:
            return detections

        keys = Counter()
        for frame_detections in self.history:
            for d in frame_detections:
                keys[(d["class"], tuple(d["bbox"]))] += 1

        threshold = (self.history.maxlen // 2) + 1
        keep = {k for k, v in keys.items() if v >= threshold}
        return [d for d in detections if (d["class"], tuple(d["bbox"])) in keep]

    def process_video_batched(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if output_path else None

        buffered = []
        detections_all = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buffered.append(frame)
            if len(buffered) >= self.batch_size:
                detections_all.extend(self._flush(buffered, writer))
                buffered = []

        if buffered:
            detections_all.extend(self._flush(buffered, writer))

        cap.release()
        if writer is not None:
            writer.release()
        return detections_all

    def _flush(self, frames, writer):
        all_detections = []
        for frame in frames:
            detections, annotated = self.detect_threats(frame)
            smoothed = self._smooth(detections)
            all_detections.extend(smoothed)

            if writer is not None and annotated is not None:
                stamp = datetime.now().strftime("%H:%M:%S")
                cv2.putText(annotated, stamp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                writer.write(annotated)

        return all_detections
