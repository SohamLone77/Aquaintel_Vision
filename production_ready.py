"""
Use pre-trained YOLOv11s directly for maritime threat detection.
Fastest path to a working system.
"""

from ultralytics import YOLO


class MaritimeThreatDetector:
    def __init__(self):
        # Use pre-trained YOLOv11s (no training needed).
        self.model = YOLO("yolo11s.pt")

        # Map COCO classes to maritime threats.
        self.threat_mapping = {
            0: {"name": "diver", "threat": "HIGH"},  # person
            9: {"name": "boat", "threat": "MEDIUM"},  # boat
            10: {"name": "ship", "threat": "LOW"},  # ship
        }

    def detect(self, image):
        results = self.model(image, conf=0.25)

        threats = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in self.threat_mapping:
                    mapping = self.threat_mapping[class_id]
                    threats.append(
                        {
                            "class": mapping["name"],
                            "confidence": float(box.conf[0]),
                            "threat_level": mapping["threat"],
                            "bbox": box.xyxy[0].tolist(),
                        }
                    )
        return threats


if __name__ == "__main__":
    detector = MaritimeThreatDetector()
    threats = detector.detect("underwater_image.jpg")
    print(threats)
