import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from production_detector import UnderwaterThreatDetector as ProductionThreatDetector
from utils.config_loader import load_runtime_config
from utils.gpu import configure_tensorflow_device

st.set_page_config(page_title="Underwater Enhancer", page_icon="🌊", layout="wide")


def list_model_files(checkpoint_dir: Path):
    model_files = list(checkpoint_dir.glob("*.keras")) + list(checkpoint_dir.glob("*.h5"))
    return sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)


def load_registry(registry_path: Path):
    if not registry_path.exists():
        return {}
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


@st.cache_resource
def load_model(model_path_str: str):
    return tf.keras.models.load_model(model_path_str, compile=False)


@st.cache_data
def get_model_input_size(model_path_str: str):
    model = tf.keras.models.load_model(model_path_str, compile=False)
    input_shape = model.input_shape
    if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 3 and input_shape[1]:
        return int(input_shape[1])
    return 128


def preprocess_image(uploaded: Image.Image, img_size: int):
    arr = np.array(uploaded.convert("RGB"))
    return preprocess_rgb_array(arr, img_size)


def preprocess_rgb_array(arr: np.ndarray, img_size: int):
    h, w = arr.shape[:2]

    # Letterbox to avoid aspect-ratio distortion before inference.
    scale = min(img_size / max(1, w), img_size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    x0 = (img_size - new_w) // 2
    y0 = (img_size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    x = canvas.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    meta = {"x0": x0, "y0": y0, "new_w": new_w, "new_h": new_h}
    return arr, x, meta


def postprocess_prediction(pred, meta: dict, output_size: tuple[int, int]):
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255.0).astype(np.uint8)

    x0 = int(meta["x0"])
    y0 = int(meta["y0"])
    new_w = int(meta["new_w"])
    new_h = int(meta["new_h"])

    cropped = pred[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)


def image_to_download_bytes(img_array: np.ndarray):
    image = Image.fromarray(img_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def fuse_details_from_original(
    original_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    detail_strength: float = 0.45,
    sharpen_amount: float = 0.4,
):
    """Recover high-frequency detail from the original while keeping enhanced color tone."""
    detail_strength = float(np.clip(detail_strength, 0.0, 1.0))
    sharpen_amount = float(np.clip(sharpen_amount, 0.0, 2.0))

    base = enhanced_rgb.astype(np.float32)
    orig = original_rgb.astype(np.float32)

    # High-frequency residual from original image.
    blur = cv2.GaussianBlur(orig, (0, 0), sigmaX=1.2, sigmaY=1.2)
    high_freq = orig - blur

    fused = base + (detail_strength * high_freq)

    if sharpen_amount > 0.0:
        # Mild unsharp mask on fused result.
        fused_blur = cv2.GaussianBlur(fused, (0, 0), sigmaX=1.0, sigmaY=1.0)
        fused = cv2.addWeighted(fused, 1.0 + sharpen_amount, fused_blur, -sharpen_amount, 0)

    return np.clip(fused, 0, 255).astype(np.uint8)


def run_inference(model, input_batch):
    pred = model.predict(input_batch, verbose=0)
    return pred[0]


def make_web_preview_video(source_path: Path):
    """Transcode to a browser-friendly H.264 MP4 when ffmpeg is available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = None

    if ffmpeg_path is None:
        return source_path, "FFmpeg not found (system or bundled). Preview may fail in some browsers for MPEG-4 output codecs."

    preview_path = source_path.with_name(f"{source_path.stem}_web.mp4")
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(preview_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    except Exception:
        return source_path, "Could not transcode preview to H.264. Downloaded video is still available."

    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path, None

    return source_path, "Preview transcode output was empty. Downloaded video is still available."


def list_yolo_model_files():
    candidates = sorted(
        Path("runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Add pre-trained fallback as a virtual option.
    return ["yolov8n.pt"] + [str(p).replace("\\", "/") for p in candidates]


class StreamlitThreatDetectorAdapter:
    """Keep Streamlit view code stable while using the production detector API."""

    def __init__(self, detector: ProductionThreatDetector):
        self.detector = detector

    def detect_threats(self, image_bgr: np.ndarray):
        detections, annotated_bgr, _ = self.detector.detect(image_bgr)

        normalized = []
        for det in detections:
            normalized.append(
                {
                    "class": det.get("class_name", "unknown"),
                    "confidence": float(det.get("confidence", 0.0)),
                    "threat_level": det.get("threat_level", "UNKNOWN"),
                    "bbox": det.get("bbox", []),
                    "timestamp": det.get("timestamp", ""),
                }
            )

        return normalized, annotated_bgr


@st.cache_resource
def load_threat_detector(
    enhancement_model_path: str,
    yolo_model_path: str,
    confidence_threshold: float,
    use_enhancement: bool,
):
    detector = ProductionThreatDetector(
        enhancement_model_path=enhancement_model_path,
        yolo_model_path=yolo_model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.45,
        enable_enhancement=use_enhancement,
    )
    return StreamlitThreatDetectorAdapter(detector)


def get_run_name_from_model_path(model_path: Path):
    stem = model_path.stem
    if stem.endswith("_best"):
        return stem[:-5]
    if stem.endswith("_final"):
        return stem[:-6]
    return stem


def infer_pipeline_type(run_name: str, run_meta: dict | None = None):
    """Infer whether a run belongs to standard or sharp pipeline."""
    lowered = run_name.lower()
    if lowered.startswith("sharp_") or "sharp" in lowered:
        return "sharp"

    if run_meta:
        cfg = run_meta.get("config", {})
        loss_type = str(cfg.get("loss_type", "")).lower()
        if loss_type == "sharp":
            return "sharp"

    return "standard"


def resolve_run_metadata(registry_data: dict, run_name: str, model_choice: Path):
    """Resolve run metadata by run key or artifact path match."""
    run_meta = registry_data.get(run_name)
    if run_meta:
        return run_meta

    model_name = model_choice.name.replace("\\", "/")
    for _, candidate in registry_data.items():
        artifacts = candidate.get("artifacts", {})
        for artifact_path in artifacts.values():
            if not artifact_path:
                continue
            artifact_name = str(artifact_path).replace("\\", "/").split("/")[-1]
            if artifact_name == model_name:
                return candidate

    return None


def build_history_fallback(run_name: str, logs_dir: Path, model_choice: Path):
    """Create minimal metadata from CSV history + file stats when registry entry is missing."""
    history_df = load_history(run_name, logs_dir)
    if history_df is None or history_df.empty:
        return None

    metrics = {
        "final_loss": float(history_df["loss"].iloc[-1]) if "loss" in history_df else None,
        "final_mae": float(history_df["mae"].iloc[-1]) if "mae" in history_df else None,
        "final_val_loss": float(history_df["val_loss"].iloc[-1]) if "val_loss" in history_df else None,
        "final_val_mae": float(history_df["val_mae"].iloc[-1]) if "val_mae" in history_df else None,
        "epochs_ran": int(len(history_df)),
        "best_val_loss": float(history_df["val_loss"].min()) if "val_loss" in history_df else None,
        "best_val_mae": float(history_df["val_mae"].min()) if "val_mae" in history_df else None,
    }

    return {
        "config": {
            "source": "history_fallback",
            "model_file": model_choice.name,
            "model_modified": model_choice.stat().st_mtime,
        },
        "metrics": metrics,
    }


def show_run_metadata(registry_data: dict, run_name: str, model_choice: Path, logs_dir: Path):
    run_meta = resolve_run_metadata(registry_data, run_name, model_choice)
    if not run_meta:
        run_meta = build_history_fallback(run_name, logs_dir, model_choice)
        if run_meta:
            st.info("Registry metadata not found. Showing metrics from training CSV fallback.")
        else:
            st.info("No registry metadata or history CSV found for selected model.")
            return

    cfg = run_meta.get("config", {})
    metrics = run_meta.get("metrics", {})
    pipeline_type = infer_pipeline_type(run_name, run_meta)

    # Backfill best validation metrics from history when registry doesn't include them.
    if metrics.get("best_val_loss") is None or metrics.get("best_val_mae") is None:
        history_df = load_history(run_name, logs_dir)
        if history_df is not None:
            if metrics.get("best_val_loss") is None and "val_loss" in history_df.columns:
                metrics["best_val_loss"] = float(history_df["val_loss"].min())
            if metrics.get("best_val_mae") is None and "val_mae" in history_df.columns:
                metrics["best_val_mae"] = float(history_df["val_mae"].min())

    left, right = st.columns(2)
    with left:
        st.markdown("### Training Config")
        st.caption(f"Pipeline Type: **{pipeline_type}**")
        st.json({
            "source": cfg.get("source", "registry"),
            "batch_size": cfg.get("batch_size"),
            "epochs": cfg.get("epochs"),
            "learning_rate": cfg.get("learning_rate"),
            "loss_type": cfg.get("loss_type"),
            "ssim_weight": cfg.get("ssim_weight"),
            "augmentation_profile": cfg.get("augmentation_profile"),
            "model_file": cfg.get("model_file", model_choice.name),
        })

    with right:
        st.markdown("### Final Metrics")
        st.json({
            "final_loss": metrics.get("final_loss"),
            "final_mae": metrics.get("final_mae"),
            "final_val_loss": metrics.get("final_val_loss"),
            "final_val_mae": metrics.get("final_val_mae"),
            "epochs_ran": metrics.get("epochs_ran"),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_val_mae": metrics.get("best_val_mae"),
        })


def load_history(run_name: str, logs_dir: Path):
    history_path = logs_dir / f"{run_name}_training.csv"
    if not history_path.exists():
        return None
    try:
        return pd.read_csv(history_path)
    except Exception:
        return None


def run_inference_view(model_choice: Path, img_size: int, registry_data: dict, logs_dir: Path):
    run_name = get_run_name_from_model_path(model_choice)

    st.markdown("### Output Detail Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        enable_detail_fusion = st.checkbox("Recover fine detail from original", value=True)
        detail_strength = st.slider("Detail strength", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    with col_b:
        enable_sharpen = st.checkbox("Apply mild sharpening", value=True)
        sharpen_amount = st.slider("Sharpen amount", min_value=0.0, max_value=1.5, value=0.40, step=0.05)

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("Image file is too large (max 20MB limit).")
            return
        
        import imghdr
        header = uploaded_file.read(512)
        uploaded_file.seek(0)
        mime = imghdr.what(None, h=header)
        if not mime or mime not in ['jpeg', 'png', 'bmp']:
            st.error(f"Invalid image format detected. Expected jpg/png/bmp but got {mime}.")
            return

    if uploaded_file is None:
        st.info("Select a model and upload an image to start inference.")
        st.markdown("### Model Metadata")
        show_run_metadata(registry_data, run_name, model_choice, logs_dir)
        return

    pil_image = Image.open(uploaded_file)
    original_arr, model_input, preprocess_meta = preprocess_image(pil_image, img_size)

    if st.button("Enhance Image", type="primary"):
        with st.spinner("Running inference..."):
            model = load_model(str(model_choice))
            prediction = run_inference(model, model_input)
            h, w = original_arr.shape[:2]
            enhanced = postprocess_prediction(prediction, preprocess_meta, (w, h))

            if enable_detail_fusion:
                enhanced = fuse_details_from_original(
                    original_rgb=original_arr,
                    enhanced_rgb=enhanced,
                    detail_strength=detail_strength,
                    sharpen_amount=(sharpen_amount if enable_sharpen else 0.0),
                )
            elif enable_sharpen and sharpen_amount > 0.0:
                enhanced_blur = cv2.GaussianBlur(enhanced.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
                enhanced = cv2.addWeighted(
                    enhanced.astype(np.float32),
                    1.0 + sharpen_amount,
                    enhanced_blur,
                    -sharpen_amount,
                    0,
                )
                enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original")
            st.image(original_arr, width="stretch")

        with col2:
            st.markdown("### Enhanced")
            st.image(enhanced, width="stretch")

        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', Path(uploaded_file.name).stem)

        st.download_button(
            label="Download enhanced PNG",
            data=image_to_download_bytes(enhanced),
            file_name=f"enhanced_{safe_name}.png",
            mime="image/png",
        )

    st.markdown("### Model Metadata")
    show_run_metadata(registry_data, run_name, model_choice, logs_dir)


def run_video_view(model_choice: Path, img_size: int):
    st.markdown("### Video Enhancement")
    st.caption("Upload a video, process it with the selected model, and download the enhanced output.")

    preset = st.selectbox(
        "Quality preset",
        options=["Fast", "Balanced", "Sharp", "Custom"],
        index=2,
        key="video_quality_preset",
    )

    preset_values = {
        "Fast": {"enable_detail_fusion": False, "detail_strength": 0.0, "sharpen_amount": 0.0},
        "Balanced": {"enable_detail_fusion": True, "detail_strength": 0.45, "sharpen_amount": 0.30},
        "Sharp": {"enable_detail_fusion": True, "detail_strength": 0.60, "sharpen_amount": 0.45},
    }

    if preset == "Custom":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            enable_detail_fusion = st.checkbox("Recover detail from original", value=True, key="video_detail_fusion")
        with col_b:
            detail_strength = st.slider(
                "Detail strength",
                min_value=0.0,
                max_value=1.0,
                value=0.60,
                step=0.05,
                key="video_detail_strength",
            )
        with col_c:
            sharpen_amount = st.slider(
                "Sharpen amount",
                min_value=0.0,
                max_value=1.5,
                value=0.45,
                step=0.05,
                key="video_sharpen_amount",
            )
    else:
        selected = preset_values[preset]
        enable_detail_fusion = selected["enable_detail_fusion"]
        detail_strength = selected["detail_strength"]
        sharpen_amount = selected["sharpen_amount"]
        st.caption(
            f"Preset settings -> detail recovery: {enable_detail_fusion}, "
            f"detail strength: {detail_strength:.2f}, sharpen: {sharpen_amount:.2f}"
        )

    max_frames = st.slider(
        "Max frames to process (0 = all)",
        min_value=0,
        max_value=3000,
        value=0,
        step=50,
        key="video_max_frames",
    )

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        if uploaded_video.size > 200 * 1024 * 1024:
            st.error("Video file is too large (max 200MB limit).")
            return
            
        header = uploaded_video.read(16)
        uploaded_video.seek(0)
        is_mp4 = b'ftyp' in header
        is_mkv = b'\x1aE\xdf\xa3' in header
        is_avi = b'RIFF' in header
        # mov is typically ftyp as well or mdat
        if not (is_mp4 or is_mkv or is_avi or b'mdat' in header):
            st.warning("Video signature may be invalid or unsupported. Proceeding with caution.")

    if uploaded_video is None:
        st.info("Upload a video file to start processing.")
        return

    st.video(uploaded_video)

    if not st.button("Enhance Video", type="primary", key="enhance_video_button"):
        return

    results_dir = Path("results/processed_videos")
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_video.name).suffix if Path(uploaded_video.name).suffix else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(uploaded_video.getbuffer())
        input_path = Path(temp_input.name)

    safe_model_name = model_choice.stem.replace(" ", "_")
    output_path = results_dir / f"{Path(uploaded_video.name).stem}_enhanced_{safe_model_name}.mp4"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_target = total_frames if max_frames == 0 else min(total_frames, max_frames)
    if frames_target <= 0:
        st.error("Video has no readable frames.")
        cap.release()
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    st.write(f"Resolution: {width} x {height} | FPS: {fps:.1f} | Frames: {frames_target}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    model = load_model(str(model_choice))

    progress = st.progress(0)
    status = st.empty()

    processed = 0
    while processed < frames_target:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_rgb, model_input, preprocess_meta = preprocess_rgb_array(frame_rgb, img_size)

        prediction = run_inference(model, model_input)
        enhanced_rgb = postprocess_prediction(prediction, preprocess_meta, (width, height))

        if enable_detail_fusion:
            enhanced_rgb = fuse_details_from_original(
                original_rgb=original_rgb,
                enhanced_rgb=enhanced_rgb,
                detail_strength=detail_strength,
                sharpen_amount=sharpen_amount,
            )

        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        writer.write(enhanced_bgr)

        processed += 1
        progress.progress(int((processed / frames_target) * 100))
        status.text(f"Processed {processed}/{frames_target} frames")

    cap.release()
    writer.release()
    try:
        input_path.unlink(missing_ok=True)
    except Exception:
        pass

    if processed == 0:
        st.error("No frames were processed.")
        return

    st.success(f"Video processing complete: {processed} frames")
    preview_path, preview_warning = make_web_preview_video(output_path)
    if preview_warning:
        st.warning(preview_warning)

    st.video(preview_path.read_bytes())

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download enhanced video",
            data=f.read(),
            file_name=output_path.name,
            mime="video/mp4",
            key="download_enhanced_video",
        )


def run_threat_detection_view(model_choice: Path):
    st.markdown("### Threat Detection (YOLO, Enhancement Optional)")
    st.caption("Run underwater threat detection on image or video. Enhancement is optional and OFF by default.")

    yolo_options = list_yolo_model_files()
    yolo_choice = st.selectbox("YOLO model", options=yolo_options, index=0, key="threat_yolo_model")

    profiles = {
        "Recall": {"conf": 0.10, "enhance": False},
        "Balanced": {"conf": 0.20, "enhance": False},
        "Strict (Recommended)": {"conf": 0.30, "enhance": False},
        "Enhancement Fallback": {"conf": 0.20, "enhance": True},
    }

    if "threat_conf" not in st.session_state:
        st.session_state["threat_conf"] = profiles["Strict (Recommended)"]["conf"]
    if "threat_use_enhancement" not in st.session_state:
        st.session_state["threat_use_enhancement"] = profiles["Strict (Recommended)"]["enhance"]

    profile_name = st.selectbox(
        "Detection profile",
        options=list(profiles.keys()),
        index=2,
        key="threat_profile",
        help="Profiles prefill confidence and enhancement settings based on validation sweeps.",
    )

    if st.session_state.get("threat_profile_applied") != profile_name:
        st.session_state["threat_conf"] = profiles[profile_name]["conf"]
        st.session_state["threat_use_enhancement"] = profiles[profile_name]["enhance"]
        st.session_state["threat_profile_applied"] = profile_name

    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, step=0.05, key="threat_conf")
    use_enhancement = st.checkbox("Apply enhancement before YOLO", key="threat_use_enhancement")

    detector = load_threat_detector(str(model_choice), yolo_choice, float(conf), bool(use_enhancement))
    mode = st.radio("Threat mode", options=["Image", "Video"], horizontal=True, key="threat_mode")

    if mode == "Image":
        uploaded_img = st.file_uploader(
            "Upload image for threat detection",
            type=["jpg", "jpeg", "png", "bmp"],
            key="threat_image_uploader",
        )

        if uploaded_img is None:
            st.info("Upload an image to run threat detection.")
            return

        img = np.array(Image.open(uploaded_img).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if st.button("Detect Threats (Image)", type="primary", key="threat_image_btn"):
            with st.spinner("Running threat detection..."):
                detections, annotated_bgr = detector.detect_threats(img_bgr)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Input")
                st.image(img, width="stretch")
            with col2:
                st.markdown("### Detection Output")
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, width="stretch")

            st.write(f"Detections found: {len(detections)}")
            if detections:
                st.dataframe(pd.DataFrame(detections), width="stretch")

            st.download_button(
                label="Download detected image",
                data=image_to_download_bytes(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)),
                file_name=f"detected_{Path(uploaded_img.name).stem}.png",
                mime="image/png",
                key="threat_image_download",
            )

        return

    uploaded_video = st.file_uploader(
        "Upload video for threat detection",
        type=["mp4", "avi", "mov", "mkv"],
        key="threat_video_uploader",
    )
    if uploaded_video is None:
        st.info("Upload a video to run threat detection.")
        return

    st.video(uploaded_video)
    max_frames = st.slider(
        "Max frames to process (0 = all)",
        min_value=0,
        max_value=3000,
        value=0,
        step=50,
        key="threat_video_max_frames",
    )

    if not st.button("Detect Threats (Video)", type="primary", key="threat_video_btn"):
        return

    out_dir = Path("results/detections")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_video.name).suffix if Path(uploaded_video.name).suffix else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_input:
        temp_input.write(uploaded_video.getbuffer())
        input_path = Path(temp_input.name)

    output_path = out_dir / f"{Path(uploaded_video.name).stem}_detected.mp4"
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_target = total if max_frames == 0 else min(total, max_frames)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    progress = st.progress(0)
    status = st.empty()

    all_detections = []
    processed = 0
    while processed < frames_target:
        ok, frame = cap.read()
        if not ok:
            break

        detections, annotated = detector.detect_threats(frame)
        writer.write(annotated)
        all_detections.extend(detections)

        processed += 1
        progress.progress(int((processed / max(1, frames_target)) * 100))
        status.text(f"Processed {processed}/{frames_target} frames")

    cap.release()
    writer.release()
    try:
        input_path.unlink(missing_ok=True)
    except Exception:
        pass

    st.success(f"Video threat detection complete: {processed} frames | detections: {len(all_detections)}")
    preview_path, preview_warning = make_web_preview_video(output_path)
    if preview_warning:
        st.warning(preview_warning)
    st.video(preview_path.read_bytes())

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download detected video",
            data=f.read(),
            file_name=output_path.name,
            mime="video/mp4",
            key="threat_video_download",
        )


def run_comparison_view(registry_data: dict, models, logs_dir: Path):
    st.markdown("### Compare Two Runs")

    run_names = sorted({*registry_data.keys(), *[get_run_name_from_model_path(p) for p in models]})
    if len(run_names) < 2:
        st.info("Need at least two runs to compare.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        run_a = st.selectbox("Run A", run_names, index=max(len(run_names) - 2, 0))
    with col_b:
        run_b = st.selectbox("Run B", run_names, index=max(len(run_names) - 1, 0))

    if run_a == run_b:
        st.warning("Choose two different runs.")
        return

    metrics_a = registry_data.get(run_a, {}).get("metrics", {})
    metrics_b = registry_data.get(run_b, {}).get("metrics", {})
    st.caption(
        f"Run A pipeline: **{infer_pipeline_type(run_a, registry_data.get(run_a))}** | "
        f"Run B pipeline: **{infer_pipeline_type(run_b, registry_data.get(run_b))}**"
    )

    rows = []
    for metric in ["final_loss", "final_mae", "final_val_loss", "final_val_mae", "epochs_ran"]:
        a = metrics_a.get(metric)
        b = metrics_b.get(metric)
        delta = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        rows.append({"metric": metric, "run_a": a, "run_b": b, "delta_b_minus_a": delta})

    st.dataframe(pd.DataFrame(rows), width="stretch")

    hist_a = load_history(run_a, logs_dir)
    hist_b = load_history(run_b, logs_dir)
    if hist_a is None or hist_b is None:
        st.info("History CSV missing for one or both runs. Metric table is still available.")
        return

    required_cols = {"epoch", "val_loss", "val_mae"}
    if not required_cols.issubset(set(hist_a.columns)) or not required_cols.issubset(set(hist_b.columns)):
        st.info("Validation columns missing in one or both history CSV files.")
        return

    # Join by epoch to safely handle runs with different lengths.
    hist_a_view = hist_a[["epoch", "val_loss", "val_mae"]].copy()
    hist_b_view = hist_b[["epoch", "val_loss", "val_mae"]].copy()
    hist_a_view.columns = ["epoch", f"{run_a}_val_loss", f"{run_a}_val_mae"]
    hist_b_view.columns = ["epoch", f"{run_b}_val_loss", f"{run_b}_val_mae"]

    chart_df = pd.merge(hist_a_view, hist_b_view, on="epoch", how="outer")
    chart_df = chart_df.sort_values("epoch").set_index("epoch")
    st.markdown("### Validation Curves")
    st.line_chart(chart_df)


def run_recommender_view(registry_data: dict, models, logs_dir: Path):
    st.markdown("### Best Run Recommender")

    run_names = sorted({*registry_data.keys(), *[get_run_name_from_model_path(p) for p in models]})
    if not run_names:
        st.info("No runs available for recommendation.")
        return

    metric_options = {
        "Best val_loss (from history CSV)": "best_val_loss",
        "Best val_mae (from history CSV)": "best_val_mae",
        "Final val_loss (from registry)": "final_val_loss",
        "Final val_mae (from registry)": "final_val_mae",
        "Final loss (from registry)": "final_loss",
        "Final mae (from registry)": "final_mae",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        metric_label = st.selectbox("Rank by metric", list(metric_options.keys()))
    with col2:
        top_k = st.slider("Top K", min_value=3, max_value=20, value=8, step=1)

    metric_key = metric_options[metric_label]
    ranking_rows = []

    for run_name in run_names:
        run_cfg = registry_data.get(run_name, {}).get("config", {})
        run_metrics = registry_data.get(run_name, {}).get("metrics", {})

        metric_value = None
        if metric_key in {"best_val_loss", "best_val_mae"}:
            history_df = load_history(run_name, logs_dir)
            if history_df is not None:
                target_col = "val_loss" if metric_key == "best_val_loss" else "val_mae"
                if target_col in history_df.columns and len(history_df[target_col].dropna()) > 0:
                    metric_value = float(history_df[target_col].min())
        else:
            raw_value = run_metrics.get(metric_key)
            if isinstance(raw_value, (int, float)):
                metric_value = float(raw_value)

        if metric_value is None:
            continue

        ranking_rows.append(
            {
                "run_name": run_name,
                "pipeline_type": infer_pipeline_type(run_name, registry_data.get(run_name)),
                "score": metric_value,
                "epochs": run_metrics.get("epochs_ran"),
                "batch_size": run_cfg.get("batch_size"),
                "learning_rate": run_cfg.get("learning_rate"),
                "augmentation_profile": run_cfg.get("augmentation_profile"),
            }
        )

    if not ranking_rows:
        st.warning("No runs had enough data to rank for the selected metric.")
        return

    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df = ranking_df.sort_values("score", ascending=True).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))

    best_row = ranking_df.iloc[0]
    st.success(
        f"Recommended run: {best_row['run_name']} (score={best_row['score']:.6f} on {metric_key})"
    )
    st.dataframe(ranking_df.head(top_k), width="stretch")


def main():
    if "health" in st.query_params:
        st.write('{"status": "ok"}')
        st.stop()

    st.title("🌊 Underwater Image Enhancement")
    st.caption("Upload an underwater image and run enhancement using a trained U-Net checkpoint.")

    device_info = configure_tensorflow_device({})
    st.info(
        f"TensorFlow device: {device_info['device']} "
        f"(GPUs: {device_info['gpu_count']}, mixed precision: {device_info['mixed_precision']})"
    )

    try:
        runtime_cfg = load_runtime_config()
    except Exception as exc:
        st.error(f"Failed to load config.yaml: {exc}")
        st.stop()

    checkpoint_dir = Path(runtime_cfg.get("checkpoint_dir", "models/checkpoints"))
    registry_path = Path(runtime_cfg.get("registry_path", "results/model_registry.json"))
    logs_dir = Path("logs/csv")

    if not checkpoint_dir.exists():
        st.error(f"Checkpoint directory not found: {checkpoint_dir}")
        st.stop()

    models = list_model_files(checkpoint_dir)
    if not models:
        st.warning("No model files found in checkpoint directory.")
        st.stop()

    registry_data = load_registry(registry_path)

    with st.sidebar:
        st.header("Settings")
        default_index = 0
        preferred = [
            i for i, p in enumerate(models)
            if ("256" in p.stem and p.suffix == ".keras" and "_final" in p.stem)
        ]
        if preferred:
            default_index = preferred[0]

        model_choice = st.selectbox(
            "Model checkpoint",
            options=models,
            index=default_index,
            format_func=lambda p: p.name,
        )
        img_size = get_model_input_size(str(model_choice))
        st.write(f"Model input size: {img_size} x {img_size}")

    tab_infer, tab_video, tab_threat, tab_compare, tab_recommend = st.tabs(
        ["Inference", "Video", "Threat Detection", "Run Comparison", "Best Run Recommender"]
    )
    with tab_infer:
        run_inference_view(model_choice=model_choice, img_size=img_size, registry_data=registry_data, logs_dir=logs_dir)
    with tab_video:
        run_video_view(model_choice=model_choice, img_size=img_size)
    with tab_threat:
        run_threat_detection_view(model_choice=model_choice)
    with tab_compare:
        run_comparison_view(registry_data=registry_data, models=models, logs_dir=logs_dir)
    with tab_recommend:
        run_recommender_view(registry_data=registry_data, models=models, logs_dir=logs_dir)


if __name__ == "__main__":
    main()
