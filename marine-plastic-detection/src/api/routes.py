from flask import Blueprint, request, jsonify
from src.models.model_loader import load_yolov5_model, load_yolov4_model
from src.detection.inference import run_inference
from src.detection.real_time_detection import start_real_time_detection

api = Blueprint('api', __name__)

yolov5_model = load_yolov5_model()
yolov4_model = load_yolov4_model()

@api.route('/api/inference/yolov5', methods=['POST'])
def yolov5_inference():
    data = request.json
    image_path = data.get('image_path')
    results = run_inference(yolov5_model, image_path)
    return jsonify(results)

@api.route('/api/inference/yolov4', methods=['POST'])
def yolov4_inference():
    data = request.json
    image_path = data.get('image_path')
    results = run_inference(yolov4_model, image_path)
    return jsonify(results)

@api.route('/api/realtime', methods=['GET'])
def realtime_detection():
    return start_real_time_detection()