#!/bin/bash
# Bikeshare Model - CI/CD gcloud Commands Reference
# Updated: Uses Artifact Registry, current gcloud syntax

# ---- Configuration ----
PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
REPO_NAME="vertex-ai-models"
IMAGE_NAME="cicd-vertex-bikeshare-model"
BUCKET_NAME="YOUR_BUCKET_NAME"

# Create Artifact Registry repository (one-time)
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --project=${PROJECT_ID}

# Submit Model Training Job to Vertex AI
gcloud ai custom-jobs create \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --worker-pool-spec=replica-count=1,machine-type='n1-standard-4',container-image-uri="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}" \
  --display-name=bike-sharing-model-training

# Upload Trained Model to Vertex AI Model Registry
gcloud ai models upload \
  --container-image-uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest" \
  --description=bikeshare-model-cicd \
  --display-name=bikeshare-model-cicd \
  --artifact-uri="gs://${BUCKET_NAME}/bike-share-rf-regression-artifact/" \
  --project=${PROJECT_ID} \
  --region=${REGION}

# Deploy Model to the Endpoint
ENDPOINT_ID="YOUR_ENDPOINT_ID"
MODEL_ID="YOUR_MODEL_ID"

gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --region=${REGION} \
  --project=${PROJECT_ID} \
  --model=${MODEL_ID} \
  --display-name=bikeshare-model-endpoint \
  --traffic-split=0=100 \
  --machine-type=n1-standard-4
