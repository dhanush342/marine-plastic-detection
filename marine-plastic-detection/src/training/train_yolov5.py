import torch
from pathlib import Path
from src.training.data_loader import load_data
from src.models.yolov5_model import YOLOv5Model

def train_yolov5():
    # Load dataset
    train_loader, valid_loader = load_data()

    # Initialize the YOLOv5 model
    model = YOLOv5Model()

    # Set training parameters
    epochs = 50
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            loss = model(images, targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    model_save_path = Path('weights/trained_models/yolov5_trained.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    train_yolov5()