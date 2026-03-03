from flask import Flask
from flask_cors import CORS
from src.api.routes import main as routes
from src.api.config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)  # Enable CORS for all routes

    app.register_blueprint(routes)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)