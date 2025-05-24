from flask import Flask, jsonify, send_from_directory
import requests
import os
from dotenv import load_dotenv

app = Flask(__name__, static_folder="../frontend")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Serve the frontend
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

# Serve static files (CSS, JS)
@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# API endpoint to fetch weather data
@app.route("/api/weather/<city>")
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else 500
        error_message = str(e) or "City not found"
        return jsonify({"error": error_message}), status_code

if __name__ == "__main__":
    app.run(port=5000, debug=True)