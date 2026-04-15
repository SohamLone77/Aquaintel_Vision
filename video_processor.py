#!/usr/bin/env python3
"""Real-time video enhancement for underwater footage."""

from __future__ import annotations

import argparse
import glob
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


DEFAULT_MODEL_GLOB = [
    "models/checkpoints/*_final.keras",
    "models/checkpoints/*_final.h5",
    "models/checkpoints/*_best.h5",
]


def configure_gpu(use_gpu: bool) -> None:
    """Configure TensorFlow runtime for GPU or CPU."""
    if not use_gpu:
        tf.config.set_visible_devices([], "GPU")
        print("[INFO] GPU disabled. Using CPU.")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[WARN] No GPU detected. Using CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[INFO] GPU acceleration enabled ({len(gpus)} GPU(s)).")


def _safe_load_model_path_from_registry(registry_path: Path) -> Optional[str]:
    """Read latest model artifact from model registry when available."""
    if not registry_path.exists():
        return None

    try:
        import json

        data = json.loads(registry_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            return None

        # Choose latest by timestamp string when available.
        sorted_items = sorted(
            data.items(),
            key=lambda item: item[1].get("timestamp", ""),
            reverse=True,
        )
        for _, record in sorted_items:
            artifacts = record.get("artifacts", {})
            for key in ["final_keras", "final_h5", "best_checkpoint"]:
                candidate = artifacts.get(key)
                if candidate and Path(candidate).exists():
                    return candidate
    except Exception:
        return None

    return None


def resolve_model_path(model_path: Optional[str]) -> str:
    """Resolve a valid model checkpoint path."""
    if model_path and Path(model_path).exists():
        return model_path

    from_registry = _safe_load_model_path_from_registry(Path("results/model_registry.json"))
    if from_registry:
        return from_registry

    for pattern in DEFAULT_MODEL_GLOB:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No model checkpoint found. Provide --model or place a checkpoint under models/checkpoints."
    )


class RealTimeVideoEnhancer:
    """Real-time enhancer for webcam, files, and RTSP streams."""

    def __init__(self, model_path: Optional[str] = None, target_size: int = 256, use_gpu: bool = True):
        print("=" * 60)
        print("REAL-TIME VIDEO ENHANCER")
        print("=" * 60)

        configure_gpu(use_gpu)

        resolved_model_path = resolve_model_path(model_path)
        print(f"[INFO] Loading model: {resolved_model_path}")

        # compile=False avoids requiring custom loss/metric symbols at inference time.
        self.model = tf.keras.models.load_model(resolved_model_path, compile=False)
        self.target_size = int(target_size)
        self.frame_count = 0

        print("[INFO] Model loaded successfully.")

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance one frame in BGR format."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        resized = cv2.resize(frame_rgb, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0

        enhanced = self.model.predict(tensor[np.newaxis, ...], verbose=0)[0]
        enhanced = np.clip(enhanced, 0.0, 1.0)
        enhanced = (enhanced * 255.0).astype(np.uint8)

        restored = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

    def process_webcam(self, camera_id: int = 0, window_name: str = "Underwater Enhancement") -> None:
        """Run webcam enhancement loop."""
        print("[INFO] Webcam mode started. Keys: q=quit, s=save, r=reset FPS")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        print(f"[INFO] Webcam: {width}x{height} @ {src_fps:.1f} FPS")

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        frame_times_ms: List[float] = []
        saved_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read webcam frame.")
                break

            start = time.time()
            enhanced = self.enhance_frame(frame)
            inference_ms = (time.time() - start) * 1000.0

            frame_times_ms.append(inference_ms)
            if len(frame_times_ms) > 30:
                frame_times_ms.pop(0)
            avg_ms = float(np.mean(frame_times_ms)) if frame_times_ms else 0.0
            avg_fps = (1000.0 / avg_ms) if avg_ms > 0.0 else 0.0

            overlay = enhanced.copy()
            info = [
                f"Enhanced FPS: {avg_fps:.1f}",
                f"Inference: {inference_ms:.1f} ms",
                f"Frame: {self.frame_count + 1}",
                f"Saved: {saved_frames}",
            ]
            for i, text in enumerate(info):
                cv2.putText(
                    overlay,
                    text,
                    (10, 30 + (i * 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            display = np.hstack([frame, overlay])
            cv2.imshow(window_name, display)
            self.frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_dir = Path("results/saved_frames")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"frame_{timestamp}.jpg"
                cv2.imwrite(str(save_path), overlay)
                saved_frames += 1
                print(f"[INFO] Saved frame: {save_path}")
            if key == ord("r"):
                frame_times_ms.clear()
                print("[INFO] FPS history reset.")

        cap.release()
        cv2.destroyAllWindows()

        final_avg_ms = float(np.mean(frame_times_ms)) if frame_times_ms else 0.0
        final_avg_fps = (1000.0 / final_avg_ms) if final_avg_ms > 0.0 else 0.0

        print("[INFO] Webcam session ended.")
        print(f"[INFO] Total frames: {self.frame_count}")
        print(f"[INFO] Saved frames: {saved_frames}")
        print(f"[INFO] Average enhanced FPS: {final_avg_fps:.1f}")

    def process_video_file(self, input_path: str, output_path: Optional[str] = None, show_preview: bool = True) -> Optional[str]:
        """Enhance a video file and write output video."""
        input_file = Path(input_path)
        if not input_file.exists():
            print(f"[ERROR] Input video not found: {input_path}")
            return None

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            print(f"[ERROR] Could not open input video: {input_path}")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            output_file = input_file.with_name(f"{input_file.stem}_enhanced{input_file.suffix}")
        else:
            output_file = Path(output_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))

        frame_times_ms: List[float] = []
        print(f"[INFO] Processing video: {input_path}")
        print(f"[INFO] Output path: {output_file}")
        print(f"[INFO] Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

        for i in tqdm(range(total_frames if total_frames > 0 else 10**9), desc="Processing", unit="frame"):
            ok, frame = cap.read()
            if not ok:
                break

            start = time.time()
            enhanced = self.enhance_frame(frame)
            frame_times_ms.append((time.time() - start) * 1000.0)

            writer.write(enhanced)

            if show_preview and (i % 15 == 0):
                preview = cv2.resize(enhanced, (max(320, width // 2), max(240, height // 2)))
                cv2.imshow("Video Processing Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Preview stop requested. Ending early.")
                    break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if not frame_times_ms:
            print("[WARN] No frames were processed.")
            return None

        avg_ms = float(np.mean(frame_times_ms))
        avg_fps = 1000.0 / avg_ms
        print(f"[INFO] Saved enhanced video: {output_file}")
        print(f"[INFO] Avg inference: {avg_ms:.1f} ms/frame")
        print(f"[INFO] Avg enhanced FPS: {avg_fps:.1f}")
        print(f"[INFO] Frames processed: {len(frame_times_ms)}")

        return str(output_file)

    def process_rtsp_stream(self, rtsp_url: str, output_path: Optional[str] = None) -> None:
        """Enhance and display RTSP stream."""
        print(f"[INFO] Connecting to RTSP stream: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("[ERROR] Could not open RTSP stream.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        writer = None
        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))
            print(f"[INFO] Recording stream to: {out_file}")

        frame_times_ms: List[float] = []
        print("[INFO] RTSP mode started. Press q to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Stream disconnected or frame read failed.")
                break

            start = time.time()
            enhanced = self.enhance_frame(frame)
            frame_times_ms.append((time.time() - start) * 1000.0)

            avg_ms = float(np.mean(frame_times_ms[-30:])) if frame_times_ms else 0.0
            avg_fps = (1000.0 / avg_ms) if avg_ms > 0.0 else 0.0
            cv2.putText(
                enhanced,
                f"Enhanced FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(enhanced)

            cv2.imshow("RTSP Enhancement", enhanced)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] RTSP session ended. Processed {len(frame_times_ms)} frames.")

    def batch_process_folder(
        self,
        input_folder: str,
        output_folder: str,
        extensions: Optional[List[str]] = None,
    ) -> List[str]:
        """Process all supported video files in a folder."""
        if extensions is None:
            extensions = [".mp4", ".avi", ".mov", ".mkv"]

        in_dir = Path(input_folder)
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            print(f"[ERROR] Input folder not found: {input_folder}")
            return []

        video_files: List[Path] = []
        for ext in extensions:
            video_files.extend(in_dir.glob(f"*{ext}"))
            video_files.extend(in_dir.glob(f"*{ext.upper()}"))

        video_files = sorted(set(video_files))
        if not video_files:
            print("[WARN] No video files found in folder.")
            return []

        print(f"[INFO] Found {len(video_files)} videos in {input_folder}")
        outputs: List[str] = []

        for file_path in video_files:
            output_path = out_dir / f"{file_path.stem}_enhanced{file_path.suffix}"
            result = self.process_video_file(str(file_path), str(output_path), show_preview=False)
            if result:
                outputs.append(result)

        print(f"[INFO] Batch complete. Processed {len(outputs)} videos.")
        return outputs


class ThreadedVideoEnhancer(RealTimeVideoEnhancer):
    """Threaded enhancer for webcam processing."""

    def __init__(self, *args, queue_size: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.processing_thread: Optional[threading.Thread] = None
        self.is_running = False

    def _processing_worker(self) -> None:
        while self.is_running:
            try:
                frame, frame_id = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            enhanced = self.enhance_frame(frame)
            try:
                self.output_queue.put((enhanced, frame_id), timeout=0.05)
            except queue.Full:
                pass

    def process_webcam_threaded(self, camera_id: int = 0) -> None:
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            self.is_running = False
            self.processing_thread.join(timeout=1.0)
            return

        frame_id = 0
        print("[INFO] Threaded webcam mode started. Press q to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            try:
                self.input_queue.put((frame, frame_id), timeout=0.01)
                frame_id += 1
            except queue.Full:
                pass

            try:
                enhanced, _ = self.output_queue.get(timeout=0.01)
                cv2.imshow("Enhanced (Threaded)", enhanced)
            except queue.Empty:
                pass

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.is_running = False
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time video enhancement for underwater footage")

    parser.add_argument("--mode", required=True, choices=["webcam", "video", "rtsp", "batch"], help="Processing mode")
    parser.add_argument("--input", type=str, help="Input video path or RTSP URL")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--target-size", type=int, default=256, help="Model input size")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--threaded", action="store_true", help="Use threaded webcam pipeline")
    parser.add_argument("--input-folder", type=str, help="Input folder for batch mode")
    parser.add_argument("--output-folder", type=str, default="results/processed_videos", help="Output folder for batch mode")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview for video mode")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    use_gpu = not args.no_gpu
    if args.threaded:
        enhancer: RealTimeVideoEnhancer = ThreadedVideoEnhancer(
            model_path=args.model,
            target_size=args.target_size,
            use_gpu=use_gpu,
        )
    else:
        enhancer = RealTimeVideoEnhancer(
            model_path=args.model,
            target_size=args.target_size,
            use_gpu=use_gpu,
        )

    if args.mode == "webcam":
        if args.threaded and isinstance(enhancer, ThreadedVideoEnhancer):
            enhancer.process_webcam_threaded(camera_id=args.camera)
        else:
            enhancer.process_webcam(camera_id=args.camera)
        return

    if args.mode == "video":
        if not args.input:
            raise ValueError("--input is required for video mode")
        enhancer.process_video_file(args.input, args.output, show_preview=not args.no_preview)
        return

    if args.mode == "rtsp":
        if not args.input:
            raise ValueError("--input is required for rtsp mode")
        enhancer.process_rtsp_stream(args.input, args.output)
        return

    if args.mode == "batch":
        if not args.input_folder:
            raise ValueError("--input-folder is required for batch mode")
        enhancer.batch_process_folder(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
