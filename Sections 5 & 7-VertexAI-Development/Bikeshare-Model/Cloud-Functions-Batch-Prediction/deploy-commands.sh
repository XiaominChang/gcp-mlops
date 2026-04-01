#!/bin/bash
# Bikeshare Model - Cloud Functions Gen 2 Deployment Commands
# Updated: Cloud Functions 2nd gen with Eventarc triggers

# ---- Configuration ----
PROJECT_ID="YOUR_PROJECT_ID"
PROJECT_NUMBER="YOUR_PROJECT_NUMBER"
REGION="us-central1"
FUNCTION_NAME="bikeshare-batch-run"
TRIGGER_BUCKET="YOUR_TRIGGER_BUCKET"  # Bucket that triggers the function

# Prerequisites - Enable required APIs
gcloud services enable \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  eventarc.googleapis.com \
  run.googleapis.com \
  --project=${PROJECT_ID}

# Grant permissions to the compute service account
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/eventarc.eventReceiver"

# Deploy Cloud Function (Gen 2) with Eventarc GCS trigger
gcloud functions deploy ${FUNCTION_NAME} \
  --gen2 \
  --runtime=python312 \
  --region=${REGION} \
  --source=. \
  --entry-point=trigger_batch_predictions \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=${TRIGGER_BUCKET}" \
  --timeout=540s \
  --memory=512Mi \
  --project=${PROJECT_ID}
