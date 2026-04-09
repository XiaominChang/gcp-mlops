#!/bin/bash
# ==============================================================================
# Cloud Build commands: manual submit and trigger setup
# ==============================================================================

# Submit a build manually using cloudbuild.yaml
gcloud builds submit --region australia-southeast1s

# Create a Cloud Build trigger connected to a GitHub repository
# Replace REPO_OWNER, REPO_NAME, and PROJECT_ID with your values.

# Step 1: Connect your GitHub repository (do this in the GCP Console first)
# Console -> Cloud Build -> Triggers -> Connect Repository

# Step 2: Create a push trigger on the main branch
gcloud builds triggers create github \
  --repo-name="REPO_NAME" \
  --repo-owner="REPO_OWNER" \
  --branch-pattern="^main$" \
  --build-config="cloudbuild.yaml" \
  --name="deploy-flask-app-on-push" \
  --region=us-central1
