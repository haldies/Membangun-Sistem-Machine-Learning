FROM python:3.10-slim

WORKDIR /app

COPY prometheus_exporter.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "prometheus_exporter.py"]
