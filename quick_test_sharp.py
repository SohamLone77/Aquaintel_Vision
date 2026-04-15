#!/usr/bin/env python3
"""Quick single-image sharpness test."""

import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def sharpen_image(image, method="kernel"):
    if method == "kernel":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return np.clip(cv2.filter2D(image, -1, kernel), 0, 255).astype(np.uint8)
    if method == "unsharp":
        blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
        return np.clip(cv2.addWeighted(image, 1.5, blurred, -0.5, 0), 0, 255).astype(np.uint8)
    return image


def main():
    print("=" * 50)
    print("QUICK SHARPENING TEST")
    print("=" * 50)

    models = glob.glob("models/checkpoints/*.h5")
    if not models:
        print("❌ No model found")
        return

    latest_model = max(models, key=os.path.getmtime)
    print(f"📂 Loading: {os.path.basename(latest_model)}")
    model = tf.keras.models.load_model(latest_model, compile=False)
    print("✅ Model loaded")
    model_input_size = int(model.input_shape[1])
    print(f"📐 Model input size detected: {model_input_size}x{model_input_size}")

    test_images = glob.glob("data/raw/*.jpg") + glob.glob("data/raw/*.png") + glob.glob("data/raw/*.jpeg")
    if not test_images:
        print("❌ No test images found")
        return

    test_img_path = test_images[0]
    print(f"📸 Testing on: {os.path.basename(test_img_path)}")

    img = cv2.imread(test_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (model_input_size, model_input_size), interpolation=cv2.INTER_LANCZOS4)

    img_norm = img_resized.astype(np.float32) / 255.0
    enhanced = model.predict(img_norm[np.newaxis, ...], verbose=0)[0]
    enhanced = (np.clip(enhanced, 0.0, 1.0) * 255).astype(np.uint8)

    enhanced_512 = cv2.resize(enhanced, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    original_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    sharp_kernel = sharpen_image(enhanced_512, "kernel")
    sharp_unsharp = sharpen_image(enhanced_512, "unsharp")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(original_512)
    axes[0, 0].set_title("Original (512x512)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced_512)
    axes[0, 1].set_title("Enhanced (512x512)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(sharp_kernel)
    axes[1, 0].set_title("Enhanced + Kernel Sharpen")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(sharp_unsharp)
    axes[1, 1].set_title("Enhanced + Unsharp Mask")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("quick_test_result.png", dpi=150)

    print("\n✅ Test complete! Results saved to: quick_test_result.png")


if __name__ == "__main__":
    main()
