import io
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from PIL import Image

app = FastAPI(title="MarineAI API")

# Load YOLOv5 model
# Using custom yolov5su.pt checkpoint
import os
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'yolov5su.pt'))
# fallback to download if missing
if not os.path.exists(model_path):
    import urllib.request
    urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov5su.pt', model_path)

# Ensure compatibility with yolov5su weights exported from v8 to v5
from ultralytics import YOLO
model = YOLO(model_path)

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    with open("interface.html", "r") as f:
        return f.read()

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/detect")
async def detect(file: UploadFile = File(...)):
    start_time = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Inference
    results = model(image)

    # Results
    detections = []
    for r in results:
        boxes = r.boxes

        # ⚡ Bolt: Batch convert tensors to lists to avoid slow element-by-element Torch tensor iteration
        if len(boxes) > 0:
            bboxes = boxes.xyxy.tolist()
            classes = boxes.cls.tolist()
            confs = boxes.conf.tolist()

            for b, c, conf in zip(bboxes, classes, confs):
                detections.append({
                    "class_id": int(c),
                    "class_name": model.names[int(c)],
                    "confidence": float(conf),
                    "bbox": b
                })

    inference_time_ms = (time.time() - start_time) * 1000

    return JSONResponse(content={
        "status": "success",
        "detections": detections,
        "inference_time_ms": inference_time_ms
    })
