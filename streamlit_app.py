import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from utils.config_loader import load_runtime_config

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


def preprocess_image(uploaded: Image.Image, img_size: int):
    arr = np.array(uploaded.convert("RGB"))
    resized = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return arr, x


def postprocess_prediction(pred):
    pred = np.clip(pred, 0.0, 1.0)
    pred = (pred * 255.0).astype(np.uint8)
    return pred


def image_to_download_bytes(img_array: np.ndarray):
    image = Image.fromarray(img_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run_inference(model, input_batch):
    pred = model.predict(input_batch, verbose=0)
    return pred[0]


def get_run_name_from_model_path(model_path: Path):
    stem = model_path.stem
    if stem.endswith("_best"):
        return stem[:-5]
    if stem.endswith("_final"):
        return stem[:-6]
    return stem


def show_run_metadata(registry_data: dict, run_name: str):
    run_meta = registry_data.get(run_name)
    if not run_meta:
        st.info("No registry metadata found for selected model.")
        return

    cfg = run_meta.get("config", {})
    metrics = run_meta.get("metrics", {})

    left, right = st.columns(2)
    with left:
        st.markdown("### Training Config")
        st.json({
            "batch_size": cfg.get("batch_size"),
            "epochs": cfg.get("epochs"),
            "learning_rate": cfg.get("learning_rate"),
            "loss_type": cfg.get("loss_type"),
            "ssim_weight": cfg.get("ssim_weight"),
            "augmentation_profile": cfg.get("augmentation_profile"),
        })

    with right:
        st.markdown("### Final Metrics")
        st.json({
            "final_loss": metrics.get("final_loss"),
            "final_mae": metrics.get("final_mae"),
            "final_val_loss": metrics.get("final_val_loss"),
            "final_val_mae": metrics.get("final_val_mae"),
            "epochs_ran": metrics.get("epochs_ran"),
        })


def load_history(run_name: str, logs_dir: Path):
    history_path = logs_dir / f"{run_name}_training.csv"
    if not history_path.exists():
        return None
    try:
        return pd.read_csv(history_path)
    except Exception:
        return None


def run_inference_view(model_choice: Path, img_size: int, registry_path: Path):
    model = load_model(str(model_choice))
    run_name = get_run_name_from_model_path(model_choice)

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is None:
        st.info("Select a model and upload an image to start inference.")
        st.markdown("### Model Metadata")
        show_run_metadata(load_registry(registry_path), run_name)
        return

    pil_image = Image.open(uploaded_file)
    original_arr, model_input = preprocess_image(pil_image, img_size)

    if st.button("Enhance Image", type="primary"):
        with st.spinner("Running inference..."):
            prediction = run_inference(model, model_input)
            enhanced = postprocess_prediction(prediction)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original")
            st.image(original_arr, width="stretch")

        with col2:
            st.markdown("### Enhanced")
            st.image(enhanced, width="stretch")

        st.download_button(
            label="Download enhanced PNG",
            data=image_to_download_bytes(enhanced),
            file_name=f"enhanced_{Path(uploaded_file.name).stem}.png",
            mime="image/png",
        )

    st.markdown("### Model Metadata")
    show_run_metadata(load_registry(registry_path), run_name)


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
    st.title("🌊 Underwater Image Enhancement")
    st.caption("Upload an underwater image and run enhancement using a trained U-Net checkpoint.")

    try:
        runtime_cfg = load_runtime_config()
    except Exception as exc:
        st.error(f"Failed to load config.yaml: {exc}")
        st.stop()

    img_size = int(runtime_cfg.get("img_size", 128))
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
        model_choice = st.selectbox(
            "Model checkpoint",
            options=models,
            format_func=lambda p: p.name,
        )
        st.write(f"Input size: {img_size} x {img_size}")

    tab_infer, tab_compare, tab_recommend = st.tabs(["Inference", "Run Comparison", "Best Run Recommender"])
    with tab_infer:
        run_inference_view(model_choice=model_choice, img_size=img_size, registry_path=registry_path)
    with tab_compare:
        run_comparison_view(registry_data=registry_data, models=models, logs_dir=logs_dir)
    with tab_recommend:
        run_recommender_view(registry_data=registry_data, models=models, logs_dir=logs_dir)


if __name__ == "__main__":
    main()
