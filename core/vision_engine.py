import cv2
from ultralytics import YOLO

class VisionAnalyzer:
    def __init__(self):
        self.model = YOLO('yolov11x.pt')  

    def analyze_scene(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % int(fps) == 0:
                results = self.model(frame)
                detections.append((frame_count/fps, results[0].boxes.cls.tolist()))
            
            frame_count += 1
        cap.release()
        return detections