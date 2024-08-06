from flask import Flask
from flask_cors import CORS

from app.routes.api import api
from app.dataService.dataService import DataService

def create_app():
    app = Flask(__name__)

    # Create DataService Intstance
    dataService = DataService()
    app.dataService = dataService

    app.register_blueprint(api, url_prefix='/api')
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    return app
