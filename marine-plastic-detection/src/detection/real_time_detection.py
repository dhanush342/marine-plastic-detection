from flask import Flask, Response
import cv2
import numpy as np
from src.models.model_loader import load_yolov5_model, load_yolov4_model
from src.detection.inference import run_inference

app = Flask(__name__)

# Load models
yolov5_model = load_yolov5_model()
yolov4_model = load_yolov4_model()

def generate_frames(model):
    camera = cv2.VideoCapture(0)  # Use the first camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Run inference
        results = run_inference(model, frame)
        
        # Process results and draw bounding boxes
        for result in results:
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Class: {cls}, Conf: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed_yolov5')
def video_feed_yolov5():
    return Response(generate_frames(yolov5_model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_yolov4')
def video_feed_yolov4():
    return Response(generate_frames(yolov4_model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)