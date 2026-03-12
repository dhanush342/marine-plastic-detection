import io
import sys
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock ultralytics YOLO before importing app
with patch('ultralytics.YOLO') as mock_yolo:
    mock_model = MagicMock()
    mock_model.names = {0: "trash_plastic"}

    # Create a mock results object
    mock_result = MagicMock()

    # Create a mock boxes object
    mock_boxes = MagicMock()
    mock_boxes.xyxy.tolist.return_value = [[10.0, 10.0, 100.0, 100.0]]
    mock_boxes.cls.tolist.return_value = [0.0]
    mock_boxes.conf.tolist.return_value = [0.95]

    mock_result.boxes = mock_boxes
    mock_model.return_value = [mock_result]

    mock_yolo.return_value = mock_model

    from backend.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_detect():
    # Create a dummy image
    img = Image.new('RGB', (200, 200), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = client.post(
        "/api/v1/detect",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "inference_time_ms" in data
    assert len(data["detections"]) == 1

    det = data["detections"][0]
    assert det["class_id"] == 0
    assert det["class_name"] == "trash_plastic"
    assert det["confidence"] == 0.95
    assert det["bbox"] == [10.0, 10.0, 100.0, 100.0]
