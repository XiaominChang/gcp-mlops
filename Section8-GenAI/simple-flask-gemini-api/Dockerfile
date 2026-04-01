FROM python:3.12-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py main.py

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
