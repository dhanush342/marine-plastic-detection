from pathlib import Path

class Config:
    """Configuration settings for the Flask application."""
    
    # Flask settings
    DEBUG = True  # Set to False in production
    TESTING = False
    
    # Model paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    YOLOV5_MODEL_PATH = BASE_DIR / 'weights' / 'yolov5su.pt'
    YOLOV4_MODEL_PATH = BASE_DIR / 'weights' / 'yolov4.pt'
    
    # Other configurations can be added here as needed
    # For example, database configurations, secret keys, etc.