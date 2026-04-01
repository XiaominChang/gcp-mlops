FROM python:3.12-slim

WORKDIR /app

COPY model-training-code.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "model-training-code.py"]
