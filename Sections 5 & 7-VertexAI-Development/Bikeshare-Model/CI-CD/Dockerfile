FROM python:3.12-slim

WORKDIR /app

COPY model_training_code.py .
COPY test-training.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "model_training_code.py"]
