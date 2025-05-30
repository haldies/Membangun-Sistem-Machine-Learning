from prometheus_client import start_http_server, Counter, Summary
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time
import threading

app = Flask(__name__)

# Load model dari MLflow
model_uri = "./model.pkl"  # Sesuaikan modelmu
model = joblib.load(model_uri)

REQUEST_COUNT = Counter(
    'prediction_requests_total', 
    'Total number of prediction requests', 
    ['endpoint', 'http_status']
)
REQUEST_LATENCY = Summary(
    'prediction_request_latency_seconds', 
    'Latency of prediction requests', 
    ['endpoint']
)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    endpoint = "/predict"
    
    try:
        data = request.json
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        http_status = "200"
        response = jsonify({"prediction": int(prediction)})
    except Exception as e:
        # Kalau error, catat http_status 500
        http_status = "500"
        response = jsonify({"error": str(e)})
        response.status_code = 500

    latency = time.time() - start_time

    # Update metrics dengan label
    REQUEST_COUNT.labels(endpoint=endpoint, http_status=http_status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    return response

def start_exporter():
    # Jalankan server metrics Prometheus di port 8000
    start_http_server(8000)
    print("Prometheus metrics exporter running on port 8000")

def start_flask():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    threading.Thread(target=start_exporter).start()
    start_flask()
