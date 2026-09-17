from ultralytics import YOLO


class Detector:

    def __init__(self):

        print("Loading YOLO26s...")

        self.model = YOLO("yolo26s.pt")

        print("YOLO26s Loaded!")

    def detect(self, frame):

        results = self.model(
            frame,
            imgsz=640,
            conf=0.45,
            verbose=False
        )

        return results