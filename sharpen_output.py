#!/usr/bin/env python3
"""Apply post-processing sharpening to model outputs."""

import glob
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from utils.gpu import configure_tensorflow_device


class ImageSharpener:
    """Collection of post-processing sharpening methods."""

    @staticmethod
    def unsharp_mask(image, sigma=1.0, strength=1.5):
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, strength, blurred, 1 - strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def kernel_sharpen(image, strength=1.0):
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ]) * strength
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def bilateral_sharpen(image):
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        return ImageSharpener.unsharp_mask(bilateral, sigma=1.0, strength=1.3)

    @staticmethod
    def laplacian_sharpen(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def adaptive_sharpen(image):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        return ImageSharpener.unsharp_mask(enhanced, sigma=0.8, strength=1.2)


def find_latest_model():
    models = glob.glob("models/checkpoints/*.h5")
    if not models:
        return None
    return max(models, key=os.path.getmtime)


def enhance_with_sharpening(model, image, method="unsharp", model_input_size=None, target_size=512):
    if model_input_size is None:
        model_input_size = int(model.input_shape[1])
    img_resized = cv2.resize(image, (model_input_size, model_input_size), interpolation=cv2.INTER_LANCZOS4)
    img_norm = img_resized.astype(np.float32) / 255.0

    enhanced = model.predict(img_norm[np.newaxis, ...], verbose=0)[0]
    enhanced_uint8 = (np.clip(enhanced, 0.0, 1.0) * 255).astype(np.uint8)

    if target_size > model_input_size:
        enhanced_uint8 = cv2.resize(
            enhanced_uint8,
            (target_size, target_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    sharpener = ImageSharpener()
    if method == "unsharp":
        return sharpener.unsharp_mask(enhanced_uint8, sigma=1.0, strength=1.5)
    if method == "kernel":
        return sharpener.kernel_sharpen(enhanced_uint8, strength=1.0)
    if method == "bilateral":
        return sharpener.bilateral_sharpen(enhanced_uint8)
    if method == "laplacian":
        return sharpener.laplacian_sharpen(enhanced_uint8)
    if method == "adaptive":
        return sharpener.adaptive_sharpen(enhanced_uint8)
    return enhanced_uint8


def create_comparison_grid(model, original_image, filename):
    methods = [
        ("Original (No Enhancement)", None),
        ("Model Only", "none"),
        ("Unsharp Mask", "unsharp"),
        ("Kernel Sharpen", "kernel"),
        ("Adaptive Sharpen", "adaptive"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for i, (title, method) in enumerate(methods):
        if method is None:
            img_resized = cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            axes[i].imshow(img_resized)
        else:
            enhanced = enhance_with_sharpening(model, original_image, method=method, target_size=512)
            axes[i].imshow(enhanced)

        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")

    plt.suptitle(f"Sharpening Comparison - {filename}", fontsize=14)
    plt.tight_layout()
    os.makedirs("results/sharp_outputs/comparison", exist_ok=True)
    plt.savefig(
        f"results/sharp_outputs/comparison/comparison_{filename}",
        dpi=150,
        bbox_inches="tight",
    )


def calculate_quality_metric(reference: np.ndarray, candidate: np.ndarray, metric: str) -> float:
    ref = reference
    cand = candidate
    if cand.shape[:2] != ref.shape[:2]:
        cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    if metric == "psnr":
        return float(psnr(ref, cand, data_range=255))

    # Default to SSIM.
    return float(ssim(ref, cand, data_range=255, channel_axis=2))


def select_best_method(
    model,
    original_image: np.ndarray,
    reference_image: np.ndarray,
    metric: str,
    target_size: int = 512,
) -> Tuple[str, np.ndarray, float, Dict[str, float]]:
    method_map = {
        "model": "none",
        "unsharp": "unsharp",
        "kernel": "kernel",
        "adaptive": "adaptive",
    }

    scores: Dict[str, float] = {}
    outputs: Dict[str, np.ndarray] = {}

    for method_name, method_type in method_map.items():
        output = enhance_with_sharpening(model, original_image, method=method_type, target_size=target_size)
        outputs[method_name] = output
        scores[method_name] = calculate_quality_metric(reference_image, output, metric)

    best_method = max(scores, key=scores.get)
    return best_method, outputs[best_method], scores[best_method], scores


def batch_sharpen(max_images=None, auto_select=False, selection_metric="ssim"):
    print("=" * 60)
    print("BATCH SHARPENING PIPELINE")
    print("=" * 60)

    device_info = configure_tensorflow_device({})
    print(
        f"\n🧠 TensorFlow device: {device_info['device']} "
        f"(GPUs: {device_info['gpu_count']}, mixed_precision: {device_info['mixed_precision']})"
    )

    model_path = find_latest_model()
    if not model_path:
        print("❌ No model found! Please train a model first.")
        return

    print(f"\n📂 Loading model: {os.path.basename(model_path)}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded")

    os.makedirs("results/sharp_outputs", exist_ok=True)
    os.makedirs("results/sharp_outputs/comparison", exist_ok=True)

    test_images = glob.glob("data/raw/*.jpg") + glob.glob("data/raw/*.png") + glob.glob("data/raw/*.jpeg")
    if not test_images:
        print("❌ No test images found in data/raw/")
        return

    if max_images is not None:
        test_images = test_images[:max_images]

    print(f"\n🖼️ Found {len(test_images)} test images")
    if auto_select:
        print(f"🎯 Auto-select mode: ON ({selection_metric.upper()})")
        print("📐 Output size: 512x512, saving only best method per image\n")
    else:
        print("📐 Output size: 512x512 with sharpening\n")

    auto_select_rows: List[Dict[str, str]] = []

    for i, img_path in enumerate(test_images):
        print(f"   [{i + 1}/{len(test_images)}] Processing: {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if auto_select:
            ref_path = os.path.join("data", "reference", os.path.basename(img_path))
            if os.path.exists(ref_path):
                ref = cv2.imread(ref_path)
                ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

                best_method, best_img, best_score, all_scores = select_best_method(
                    model,
                    img_rgb,
                    ref_rgb,
                    metric=selection_metric,
                    target_size=512,
                )

                output_path = f"results/sharp_outputs/best_{os.path.basename(img_path)}"
                cv2.imwrite(output_path, cv2.cvtColor(best_img, cv2.COLOR_RGB2BGR))

                auto_select_rows.append({
                    "image": os.path.basename(img_path),
                    "metric": selection_metric,
                    "best_method": best_method,
                    "best_score": f"{best_score:.6f}",
                    "model_score": f"{all_scores['model']:.6f}",
                    "unsharp_score": f"{all_scores['unsharp']:.6f}",
                    "kernel_score": f"{all_scores['kernel']:.6f}",
                    "adaptive_score": f"{all_scores['adaptive']:.6f}",
                })
            else:
                fallback = enhance_with_sharpening(model, img_rgb, method="none", target_size=512)
                output_path = f"results/sharp_outputs/best_{os.path.basename(img_path)}"
                cv2.imwrite(output_path, cv2.cvtColor(fallback, cv2.COLOR_RGB2BGR))

                auto_select_rows.append({
                    "image": os.path.basename(img_path),
                    "metric": selection_metric,
                    "best_method": "model",
                    "best_score": "NA",
                    "model_score": "NA",
                    "unsharp_score": "NA",
                    "kernel_score": "NA",
                    "adaptive_score": "NA",
                })
        else:
            methods = {
                "original": "none",
                "unsharp": "unsharp",
                "kernel": "kernel",
                "adaptive": "adaptive",
            }

            for method_name, method_type in methods.items():
                enhanced = enhance_with_sharpening(model, img_rgb, method=method_type, target_size=512)
                output_path = f"results/sharp_outputs/{method_name}_{os.path.basename(img_path)}"
                cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))

        if i == 0:
            create_comparison_grid(model, img_rgb, os.path.basename(img_path))

    if auto_select and auto_select_rows:
        summary_path = f"results/sharp_outputs/auto_select_{selection_metric}.csv"
        with open(summary_path, "w", encoding="utf-8") as fp:
            fp.write(
                "image,metric,best_method,best_score,model_score,unsharp_score,kernel_score,adaptive_score\n"
            )
            for row in auto_select_rows:
                fp.write(
                    f"{row['image']},{row['metric']},{row['best_method']},{row['best_score']},"
                    f"{row['model_score']},{row['unsharp_score']},{row['kernel_score']},{row['adaptive_score']}\n"
                )
        print(f"📊 Auto-select summary saved: {summary_path}")

    print("\n✅ All images processed!")
    print("📁 Results saved to: results/sharp_outputs/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch sharpen model outputs")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Save only the best method per image using reference metrics",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["psnr", "ssim"],
        default="ssim",
        help="Metric used for best-method selection when --auto-select is enabled",
    )
    args = parser.parse_args()

    batch_sharpen(
        max_images=args.max_images,
        auto_select=args.auto_select,
        selection_metric=args.selection_metric,
    )
