FROM python:3.12-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py main.py

# This can be run as a batch job or a web service
# For batch: docker run <image>
# For web service: uncomment the CMD below and add Flask endpoints
CMD ["python", "main.py"]
