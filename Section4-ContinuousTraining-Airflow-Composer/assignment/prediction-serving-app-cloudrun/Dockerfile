FROM python:3.12-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY main.py main.py
COPY model.joblib model.joblib
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
