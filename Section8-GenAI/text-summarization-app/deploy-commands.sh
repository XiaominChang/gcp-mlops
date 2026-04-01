#!/bin/bash
# Manual deployment commands for the document summarization Gemini app
# Update PROJECT_ID with your Google Cloud project ID

PROJECT_ID="your-project-id"
REGION="us-central1"
REPO_NAME="gemini-apps"
IMAGE_NAME="llm-summarize-word-docs"
AR_HOSTNAME="us-central1-docker.pkg.dev"

# Step 1: Create Artifact Registry repository (skip if already created)
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Gemini apps" \
    2>/dev/null || echo "Repository already exists"

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

curl -X POST "$SERVICE_URL/summarize_word_documents" \
    -H "Content-Type: application/json" \
    -d '{"file_name": "future_of_ai.docx"}'

echo ""

curl -X POST "$SERVICE_URL/summarize_word_documents" \
    -H "Content-Type: application/json" \
    -d '{"file_name": "diet-nutritions.docx"}'
