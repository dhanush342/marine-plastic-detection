from pathlib import Path
import torch

class ModelLoader:
    def __init__(self, model_type='yolov5'):
        self.model_type = model_type
        self.model = self.load_model()

    def load_model(self):
        model_path = self.get_model_path()
        if self.model_type == 'yolov5':
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        elif self.model_type == 'yolov4':
            model = torch.hub.load('AlexeyAB/darknet', 'custom', path=model_path, source='github')
        else:
            raise ValueError("Model type must be 'yolov5' or 'yolov4'")
        return model

    def get_model_path(self):
        weights_dir = Path(__file__).resolve().parent.parent.parent / 'weights'
        if self.model_type == 'yolov5':
            return weights_dir / 'yolov5su.pt'
        elif self.model_type == 'yolov4':
            return weights_dir / 'yolov4.pt'
        else:
            raise ValueError("Model type must be 'yolov5' or 'yolov4'")

    def get_model(self):
        return self.model

# Example usage:
# loader = ModelLoader(model_type='yolov5')
# model = loader.get_model()