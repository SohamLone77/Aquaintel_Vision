"""Interactive YOLO label verification utility."""

from __future__ import annotations

import glob
import os

import cv2


class LabelVerifier:
    def __init__(self, image_dir="underwater_dataset/images/train", label_dir="underwater_dataset/labels/train"):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.classes = {
            0: "diver",
            1: "suspicious_object",
            2: "boat",
            3: "ship",
            4: "submarine",
            5: "pipeline",
            6: "obstacle",
            7: "marine_life",
            8: "swimmer",
            9: "underwater_vehicle",
        }

    def verify_all(self):
        images = glob.glob(f"{self.image_dir}/*.jpg") + glob.glob(f"{self.image_dir}/*.png")
        print(f"Verifying {len(images)} images")
        print("Controls: SPACE=verified | ESC=skip | q=quit")

        verified, skipped = [], []
        for img_path in images:
            label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt").replace(".png", ".txt"))
            if not os.path.exists(label_path):
                skipped.append(img_path)
                continue

            ok = self.verify_single(img_path, label_path)
            (verified if ok else skipped).append(img_path)

        print(f"Verified: {len(verified)} | Skipped: {len(skipped)}")
        return verified, skipped

    def verify_single(self, image_path, label_path):
        img = cv2.imread(image_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_w = float(parts[3]) * w
                box_h = float(parts[4]) * h
                x1, y1 = int(x_center - box_w / 2), int(y_center - box_h / 2)
                x2, y2 = int(x_center + box_w / 2), int(y_center + box_h / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, self.classes.get(class_id, str(class_id)), (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Label Verification", img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 32:
                cv2.destroyAllWindows()
                return True
            if key == 27:
                cv2.destroyAllWindows()
                return False
            if key == ord("q"):
                cv2.destroyAllWindows()
                raise SystemExit(0)


if __name__ == "__main__":
    LabelVerifier().verify_all()
