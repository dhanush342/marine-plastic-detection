from flask import jsonify
import cv2
import numpy as np
from src.models.model_loader import load_yolov5_model, load_yolov4_model

# Load models
yolov5_model = load_yolov5_model()
yolov4_model = load_yolov4_model()

def run_inference(image):
    # Preprocess the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Resize to model input size
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference with YOLOv5
    yolov5_results = yolov5_model.predict(img)
    
    # Run inference with YOLOv4
    yolov4_results = yolov4_model.predict(img)

    # Post-process results
    yolov5_detections = post_process(yolov5_results)
    yolov4_detections = post_process(yolov4_results)

    return {
        "yolov5": yolov5_detections,
        "yolov4": yolov4_detections
    }

def post_process(results):
    detections = []
    for result in results:
        for detection in result:
            x1, y1, x2, y2, conf, cls = detection
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": int(cls)
            })
    return detections

def inference_from_image(image_path):
    image = cv2.imread(image_path)
    return run_inference(image)

def inference_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = run_inference(frame)
        results.append(result)
    
    cap.release()
    return results