"""
YOLO-World Open-Vocabulary Detector Module
Wraps Ultralytics YOLO-World v2 to enable real-time detection of arbitrary
recipe ingredients using pre-computed CLIP text embeddings on Apple Silicon MPS.
"""

from typing import List, Dict, Tuple, Optional
import torch
from ultralytics import YOLO
import numpy as np


class YOLOWorldDetector:
    """
    Open-vocabulary object detector using YOLO-World v2.
    Pre-computes and caches text embeddings for target ingredients once,
    enabling high-throughput visual detection on live video frames.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-worldv2.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        imgsz: int = 640
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        # Auto-detect Apple Silicon MPS or fall back to CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[Detector] Loading {model_name} on device '{self.device}'...")
        self.model = YOLO(model_name)

        # Pre-fuse Conv+BatchNorm on CPU to prevent PyTorch MPS fusion crash on Apple Silicon
        try:
            self.model.fuse()
        except Exception:
            pass

        self.current_classes: List[str] = []

    def set_classes(self, target_ingredients: List[str]):
        """
        Encodes target ingredient names into offline text embeddings
        and assigns them as fixed classifier weights in the detection head.
        This must be called once per recipe before processing frames.
        """
        if not target_ingredients:
            print("[Detector] Warning: No target ingredients provided.")
            return

        # Normalize strings
        cleaned = [item.strip() for item in target_ingredients if item.strip()]
        self.current_classes = cleaned
        print(f"[Detector] Setting {len(cleaned)} open-vocabulary target classes: {cleaned}")
        self.model.set_classes(cleaned)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Runs object detection on a single video frame.
        Returns a list of detected objects:
            [{"name": str, "confidence": float, "box": (x1, y1, x2, y2)}]
        """
        if frame is None or len(self.current_classes) == 0:
            return []

        # Predict with letterboxing to self.imgsz
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_threshold,
            verbose=False
        )

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                # Resolve class name
                if cls_id in self.model.names:
                    class_name = self.model.names[cls_id]
                elif cls_id < len(self.current_classes):
                    class_name = self.current_classes[cls_id]
                else:
                    class_name = f"item_{cls_id}"

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    "name": class_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                })

        return detections


if __name__ == "__main__":
    print("Testing YOLOWorldDetector initialization...")
    detector = YOLOWorldDetector()
    detector.set_classes(["apple", "banana", "bottle"])
    # Dummy frame test
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect(dummy)
    print(f"Detection test completed on dummy frame: {len(res)} detections.")

