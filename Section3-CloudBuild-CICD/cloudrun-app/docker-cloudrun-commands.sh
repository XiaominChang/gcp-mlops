# Step-1
docker build -t demo-flask-app .

# Push to Container Registry 
# docker tag demo-flask-app gcr.io/udemy-mlops-395416/demo-flask-app
# docker push gcr.io/udemy-mlops-395416/demo-flask-app

# gcloud run deploy demo-flask-app --image gcr.io/udemy-mlops-395416/demo-flask-app --region us-central1


# Push to Artifact Registry 
docker tag demo-flask-app australia-southeast1-docker.pkg.dev/udemy-mlops-492103/python-apps/demo-flask-app
docker push australia-southeast1-docker.pkg.dev/udemy-mlops-492103/python-apps/demo-flask-app

gcloud run deploy demo-flask-app \
--image australia-southeast1-docker.pkg.dev/udemy-mlops-492103/python-apps/demo-flask-app \
--region australia-southeast1
#!/bin/bash
# ==============================================================================
# Deploy Flask App to Artifact Registry and Cloud Run
# ==============================================================================
# Replace PROJECT_ID and REPO with your values before running.
# ==============================================================================

PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
REPO="python-apps"
IMAGE="demo-flask-app"

# Step 1: Configure Docker for Artifact Registry
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Step 2: Create Artifact Registry repository (if not exists)
gcloud artifacts repositories create ${REPO} \
  --repository-format=docker \
  --location=${REGION} \
  --description="Python application Docker images"

# Step 3: Build Docker image
docker build -t ${IMAGE} .

# Step 4: Tag for Artifact Registry
docker tag ${IMAGE} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}

# Step 5: Push to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}

# Step 6: Deploy to Cloud Run
gcloud run deploy ${IMAGE} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE} \
  --region ${REGION} \
  --allow-unauthenticated
