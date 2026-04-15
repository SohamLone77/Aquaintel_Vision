#!/usr/bin/env python3
"""Quick smoke test for video enhancement pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from video_processor import RealTimeVideoEnhancer


def create_test_video(output_path: str = "test_underwater_video.mp4", fps: int = 24, duration_s: int = 4) -> str:
    """Create a small synthetic underwater-like sample video."""
    width, height = 640, 480
    frame_count = fps * duration_s

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Simulate underwater cast + moving highlights.
        frame[:, :, 0] = np.random.randint(70, 160, (height, width), dtype=np.uint8)
        frame[:, :, 1] = np.random.randint(40, 110, (height, width), dtype=np.uint8)
        frame[:, :, 2] = np.random.randint(10, 70, (height, width), dtype=np.uint8)

        center_x = int((i / max(1, frame_count - 1)) * (width - 1))
        cv2.circle(frame, (center_x, height // 2), 40, (220, 180, 90), -1)

        noise = np.random.randint(0, 25, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        writer.write(frame)

    writer.release()
    print(f"[INFO] Test video created: {output_path}")
    return output_path


def main() -> None:
    test_video = create_test_video()

    enhancer = RealTimeVideoEnhancer(model_path=None, target_size=256, use_gpu=True)
    out_path = enhancer.process_video_file(test_video, "test_enhanced.mp4", show_preview=False)

    if out_path and Path(out_path).exists():
        print(f"[INFO] Test complete. Enhanced video: {out_path}")
    else:
        print("[WARN] Test ended but no output file was generated.")


if __name__ == "__main__":
    main()
