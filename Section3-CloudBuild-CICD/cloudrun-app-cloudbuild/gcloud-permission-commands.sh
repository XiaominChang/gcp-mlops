#!/bin/bash
# ==============================================================================
# Grant Cloud Build service account the required IAM roles
# ==============================================================================
# Replace PROJECT_ID and PROJECT_NUMBER with your values.
# Find your project number: gcloud projects describe PROJECT_ID --format='value(projectNumber)'
# ==============================================================================

PROJECT_ID="YOUR_PROJECT_ID"
PROJECT_NUMBER="YOUR_PROJECT_NUMBER"
SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

# Assign Service Account User role (required to deploy to Cloud Run)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member=serviceAccount:${SA} \
  --role=roles/iam.serviceAccountUser

# Assign Cloud Run Admin role (required to create/update Cloud Run services)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member=serviceAccount:${SA} \
  --role=roles/run.admin

# Assign Artifact Registry Writer role (required to push images)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member=serviceAccount:${SA} \
  --role=roles/artifactregistry.writer
