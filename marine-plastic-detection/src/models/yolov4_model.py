from pathlib import Path
import torch
import cv2

class YOLOv4Model:
    def __init__(self, weights_path):
        self.model = self.load_model(weights_path)

    def load_model(self, weights_path):
        model = torch.hub.load('ultralytics/yolov4', 'custom', path=weights_path, force_reload=True)
        return model

    def infer(self, img):
        results = self.model(img)
        return results

    def process_frame(self, frame):
        results = self.infer(frame)
        detections = results.xyxy[0]  # Get detections
        return detections

    def draw_detections(self, frame, detections):
        for *box, conf, cls in detections.tolist():
            x1, y1, x2, y2 = map(int, box)
            label = f'Class: {int(cls)}, Conf: {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

def main():
    weights_path = str(Path(__file__).resolve().parent.parent.parent / 'weights' / 'yolov4.pt')
    yolo_model = YOLOv4Model(weights_path)

    # Example of using the model for inference on a video stream
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_model.process_frame(frame)
        frame = yolo_model.draw_detections(frame, detections)

        cv2.imshow('YOLOv4 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()