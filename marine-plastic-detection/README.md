# Marine Plastic Detection Project

This project aims to detect marine plastic using state-of-the-art object detection models, specifically YOLOv4 and YOLOv5. The application leverages real-time video processing and provides a web interface for user interaction.

## Project Structure

```
marine-plastic-detection
├── src
│   ├── api
│   │   ├── app.py               # Main entry point for the Flask application
│   │   ├── routes.py            # API endpoints for model inference and real-time detection
│   │   └── config.py            # Configuration settings for the Flask application
│   ├── models
│   │   ├── yolov5_model.py      # Implementation details for the YOLOv5 model
│   │   ├── yolov4_model.py      # Implementation details for the YOLOv4 model
│   │   └── model_loader.py      # Handles loading trained models from the weights directory
│   ├── detection
│   │   ├── real_time_detection.py # Logic for real-time detection using loaded models
│   │   ├── inference.py         # Functions for running inference on images or video frames
│   │   └── utils.py             # Utility functions for preprocessing and post-processing
│   └── training
│       ├── train_yolov5.py      # Training script for the YOLOv5 model
│       ├── train_yolov4.py      # Training script for the YOLOv4 model
│       └── data_loader.py       # Handles loading datasets from the data directory
├── frontend
│   ├── interface.html           # Main HTML interface for the web application
│   ├── static
│   │   ├── css
│   │   │   └── style.css        # CSS styles for the frontend interface
│   │   └── js
│   │       └── script.js        # JavaScript code for user interactions and API calls
│   └── templates                # Directory for additional HTML templates
├── data
│   ├── train                    # Directory for training images and annotations
│   ├── valid                    # Directory for validation images and annotations
│   ├── test                     # Directory for test images and annotations
│   └── data.yaml                # Configuration details for the dataset
├── weights
│   ├── yolov5su.pt             # Pre-trained weights for the YOLOv5 model
│   ├── yolov4.pt               # Pre-trained weights for the YOLOv4 model
│   └── trained_models           # Directory to store trained models
├── requirements.txt             # Lists Python dependencies required for the project
├── .gitignore                   # Specifies files and directories to ignore by Git
└── README.md                    # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/dhanush342/marine-plastic-detection.git
   cd marine-plastic-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Training the Models**:
   - To train the YOLOv5 model, run:
     ```
     python src/training/train_yolov5.py
     ```
   - To train the YOLOv4 model, run:
     ```
     python src/training/train_yolov4.py
     ```

2. **Running the Flask Application**:
   - Start the Flask server:
     ```
     python src/api/app.py
     ```

3. **Accessing the Web Interface**:
   - Open your web browser and navigate to `http://localhost:5000` to access the interface.

## Real-Time Detection

The application supports real-time detection using a webcam or video feed. Users can upload images or stream video to detect marine plastic in real-time.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.