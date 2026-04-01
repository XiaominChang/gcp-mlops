#!/bin/bash
# ==============================================================================
# Deploy and test the Bikeshare RandomForest model
# ==============================================================================

PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
REPO="ml-models"
IMAGE="bikeshare-model"

# Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Create Artifact Registry repository (if not exists)
gcloud artifacts repositories create ${REPO} \
  --repository-format=docker \
  --location=${REGION} \
  --description="ML model Docker images" 2>/dev/null || true

# Build Docker image
docker build -t ${IMAGE} .

# Tag for Artifact Registry
docker tag ${IMAGE} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}

# Push to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}

# Deploy to Cloud Run
gcloud run deploy ${IMAGE} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE} \
  --region ${REGION} \
  --allow-unauthenticated

# Submit via Cloud Build
# gcloud builds submit --region ${REGION}

# ==============================================================================
# Test locally (flask app running on port 5052)
# ==============================================================================

# Health check
curl http://127.0.0.1:5052/health

# Predict: winter morning, cold, weekend
curl -X POST http://127.0.0.1:5052/predict \
-H "Content-Type: application/json" \
-d '{"temp":0.24,"humidity":0.81,"season_2":0,"season_3":0,"season_4":0,"month_2":0,"month_3":0,"month_4":0,"month_5":0,"month_6":0,"month_7":0,"month_8":0,"month_9":0,"month_10":0,"month_11":0,"month_12":0,"hour_1":0,"hour_2":0,"hour_3":0,"hour_4":0,"hour_5":0,"hour_6":0,"hour_7":0,"hour_8":0,"hour_9":0,"hour_10":0,"hour_11":0,"hour_12":0,"hour_13":0,"hour_14":0,"hour_15":0,"hour_16":0,"hour_17":0,"hour_18":0,"hour_19":0,"hour_20":0,"hour_21":0,"hour_22":0,"hour_23":0,"holiday_1":0,"weekday_1":0,"weekday_2":0,"weekday_3":0,"weekday_4":0,"weekday_5":0,"weekday_6":1,"workingday_1":0,"weather_2":0,"weather_3":0,"weather_4":0}'

# ==============================================================================
# Test Cloud Run deployment (replace URL with your service URL)
# ==============================================================================
# curl -X POST https://bikeshare-model-XXXXX-uc.a.run.app/predict \
# -H "Content-Type: application/json" \
# -d '{ ... same JSON as above ... }'
