from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path, confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        return results[0]