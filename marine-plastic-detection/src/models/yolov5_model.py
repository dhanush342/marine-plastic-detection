from pathlib import Path
import torch

class YOLOv5Model:
    def __init__(self, model_path='weights/yolov5su.pt'):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
        model.eval()
        return model

    def infer(self, img):
        results = self.model(img)
        return results

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_saved_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()

# Example usage:
# if __name__ == "__main__":
#     yolo_model = YOLOv5Model()
#     img = 'path/to/image.jpg'  # Replace with your image path
#     results = yolo_model.infer(img)
#     print(results)