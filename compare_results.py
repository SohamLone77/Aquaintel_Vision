#!/usr/bin/env python3
"""Compare original, enhanced, and sharpened results with PSNR/SSIM metrics."""

import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from sharpen_output import enhance_with_sharpening


def calculate_metrics(original, enhanced):
    if original.max() <= 1.0:
        original = (original * 255).astype(np.uint8)
    if enhanced.max() <= 1.0:
        enhanced = (enhanced * 255).astype(np.uint8)

    psnr_val = psnr(original, enhanced, data_range=255)
    ssim_val = ssim(original, enhanced, data_range=255, channel_axis=2)
    return psnr_val, ssim_val


def compare_all_methods(max_images=3):
    print("=" * 60)
    print("COMPARING ENHANCEMENT METHODS")
    print("=" * 60)

    test_images = glob.glob("data/raw/*.jpg") + glob.glob("data/raw/*.png") + glob.glob("data/raw/*.jpeg")
    if not test_images:
        print("❌ No test images found")
        return

    models = glob.glob("models/checkpoints/*.h5")
    if not models:
        print("❌ No model found")
        return

    model_path = max(models, key=os.path.getmtime)
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"✅ Model loaded: {os.path.basename(model_path)}")

    os.makedirs("results", exist_ok=True)
    all_metrics = []
    has_reference = False

    for img_path in test_images[:max_images]:
        print(f"\n📸 Analyzing: {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ref_path = os.path.join("data", "reference", os.path.basename(img_path))
        if os.path.exists(ref_path):
            ref = cv2.imread(ref_path)
            ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
            has_reference = True
        else:
            ref_rgb = None

        versions = {
            "Original": img_rgb,
            "Model Only": enhance_with_sharpening(model, img_rgb, method="none", target_size=512),
            "Unsharp Mask": enhance_with_sharpening(model, img_rgb, method="unsharp", target_size=512),
            "Kernel Sharpen": enhance_with_sharpening(model, img_rgb, method="kernel", target_size=512),
            "Adaptive": enhance_with_sharpening(model, img_rgb, method="adaptive", target_size=512),
        }

        metrics = {}
        for name, img_version in versions.items():
            if ref_rgb is not None:
                img_resized = cv2.resize(img_version, (ref_rgb.shape[1], ref_rgb.shape[0]))
                psnr_val, ssim_val = calculate_metrics(ref_rgb, img_resized)
                metrics[name] = {"PSNR": psnr_val, "SSIM": ssim_val}

        all_metrics.append(metrics)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, (title, img_ver) in enumerate(versions.items()):
            row, col = i // 3, i % 3
            axes[row, col].imshow(img_ver)
            if ref_rgb is not None and title in metrics:
                axes[row, col].set_title(
                    f"{title}\nPSNR: {metrics[title]['PSNR']:.2f} dB\nSSIM: {metrics[title]['SSIM']:.3f}",
                    fontsize=10,
                )
            else:
                axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis("off")

        if ref_rgb is not None:
            axes[1, 2].imshow(ref_rgb)
            axes[1, 2].set_title("Reference (Ground Truth)", fontsize=10)
            axes[1, 2].axis("off")

        plt.suptitle(f"Comparison - {os.path.basename(img_path)}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"results/comparison_{os.path.basename(img_path).split('.')[0]}.png", dpi=150)

    print("\n" + "=" * 60)
    print("SUMMARY - AVERAGE METRICS")
    print("=" * 60)

    if all_metrics and has_reference and all_metrics[0]:
        method_names = list(all_metrics[0].keys())
        print(f"\n{'Method':<20} {'PSNR (dB)':<12} {'SSIM':<10}")
        print("-" * 42)

        for method in method_names:
            psnr_vals = [m[method]["PSNR"] for m in all_metrics if method in m]
            ssim_vals = [m[method]["SSIM"] for m in all_metrics if method in m]
            if psnr_vals:
                print(f"{method:<20} {np.mean(psnr_vals):<12.2f} {np.mean(ssim_vals):<10.4f}")
    else:
        print("No reference metrics were computed (matching files in data/reference not found).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare enhancement methods")
    parser.add_argument("--max-images", type=int, default=3, help="Number of test images to compare")
    args = parser.parse_args()

    compare_all_methods(max_images=args.max_images)
