#!/bin/bash
# ==============================================================================
# Deploy and test the XGBoost Coupon Recommendation model
# ==============================================================================

PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
REPO="ml-models"
IMAGE="xgboost-coupon-model"

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

# Submit Cloud Build
gcloud builds submit --region ${REGION}

# ==============================================================================
# Test locally (flask app running on port 5051)
# ==============================================================================
curl -X POST http://127.0.0.1:5051/predict \
-H "Content-Type: application/json" \
-d '{
     "destination": "No Urgent Place",
     "passanger": "Kid(s)",
     "weather": "Sunny",
     "temperature": 80,
     "time": "10AM",
     "coupon": "Bar",
     "expiration": "1d",
     "gender": "Female",
     "age": "21",
     "maritalStatus": "Unmarried partner",
     "has_children": 1,
     "education": "Some college - no degree",
     "occupation": "Unemployed",
     "income": "$37500 - $49999",
     "Bar": "never",
     "CoffeeHouse": "never",
     "CarryAway": "4~8",
     "RestaurantLessThan20": "4~8",
     "Restaurant20To50": "1~3",
     "toCoupon_GEQ15min": 1,
     "toCoupon_GEQ25min": 0,
     "direction_same": 0
}'

# ==============================================================================
# Test Cloud Run deployment (replace URL with your Cloud Run service URL)
# ==============================================================================
# curl -X POST https://xgboost-coupon-model-XXXXX-uc.a.run.app/predict \
# -H "Content-Type: application/json" \
# -d '{ ... same JSON as above ... }'
