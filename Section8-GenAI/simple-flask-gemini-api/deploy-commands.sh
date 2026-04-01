#!/bin/bash
# Manual deployment commands for the text classification Gemini app
# Update PROJECT_ID with your Google Cloud project ID

PROJECT_ID="your-project-id"
REGION="us-central1"
REPO_NAME="gemini-apps"
IMAGE_NAME="llm-text-classification"
AR_HOSTNAME="us-central1-docker.pkg.dev"

# Step 1: Create Artifact Registry repository (one-time setup)
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Gemini apps"

# Step 2: Configure Docker to authenticate with Artifact Registry
gcloud auth configure-docker $AR_HOSTNAME

# Step 3: Build the Docker image
docker build -t $IMAGE_NAME .

# Step 4: Tag the image for Artifact Registry
docker tag $IMAGE_NAME $AR_HOSTNAME/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest

# Step 5: Push to Artifact Registry
docker push $AR_HOSTNAME/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest

# Step 6: Deploy to Cloud Run
gcloud run deploy $IMAGE_NAME \
    --image $AR_HOSTNAME/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 512Mi

# Step 7: Test the deployed service
SERVICE_URL=$(gcloud run services describe $IMAGE_NAME --region $REGION --format 'value(status.url)')

echo "Service URL: $SERVICE_URL"

curl -X POST "$SERVICE_URL/simple_classification" \
    -H "Content-Type: application/json" \
    -d '{"msg": "Im wondering where to travel next"}'

echo ""

curl -X POST "$SERVICE_URL/simple_classification_with_exp" \
    -H "Content-Type: application/json" \
    -d '{"msg": "Im wondering where to travel next"}'

echo ""

curl -X POST "$SERVICE_URL/simple_classification_with_exp" \
    -H "Content-Type: application/json" \
    -d '{"msg": "i was upset with the performance ratings after having worked so hard"}'
