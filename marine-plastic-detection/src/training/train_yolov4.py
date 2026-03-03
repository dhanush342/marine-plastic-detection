import torch
from models.yolov4_model import YOLOv4
from training.data_loader import load_data
import os

def train_yolov4():
    # Load the dataset
    train_loader, valid_loader = load_data()

    # Initialize the YOLOv4 model
    model = YOLOv4()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = model.loss_function

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    model_save_path = os.path.join('weights', 'trained_models', 'yolov4_trained.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    train_yolov4()