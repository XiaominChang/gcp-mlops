#!/bin/bash
# ==============================================================================
# Deploy and test the XGBoost Coupon Recommendation model
# ==============================================================================

PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
REPO="ml-models"
IMAGE="xgboost-coupon-model"

# Build Docker image
docker build --platform linux/amd64 -t ${IMAGE} .
docker buildx build --platform linux/amd64 -t xgboost-coupon-model:v2 .

# Tag for Artifact Registry
docker tag ${IMAGE} ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}
docker tag xgboost-coupon-model:v2 australia-southeast1-docker.pkg.dev/udemy-mlops-492103/ml-models/xgboost-coupon-model:v2

# Push to Artifact Registry
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}
docker push australia-southeast1-docker.pkg.dev/udemy-mlops-492103/ml-models/xgboost-coupon-model:v2

# Deploy to Cloud Run
gcloud run deploy ${IMAGE} \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE} \
  --region ${REGION} \
  --allow-unauthenticated

gcloud run deploy xgboost-coupon-model \
  --image australia-southeast1-docker.pkg.dev/udemy-mlops-492103/ml-models/xgboost-coupon-model:v2 \
  --region australia-southeast1 \
  --allow-unauthenticated


gcloud run revisions list --service xgboost-coupon-model --region australia-southeast1
gcloud run services update-traffic xgboost-coupon-model --to-revisions=xgboost-coupon-model-00001-5tl=90,xgboost-coupon-model-00004-z46=10 --region australia-southeast1

# Submit Cloud Build
gcloud builds submit --region ${REGION}

# ==============================================================================
# Test locally (flask app running on port 5051)
# ==============================================================================
curl -X POST https://xgboost-coupon-model-188673622020.australia-southeast1.run.app/predict \
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

gcloud run services add-iam-policy-binding xgboost-coupon-model \
  --region=australia-southeast1 \
  --member="allUsers" \
  --role="roles/run.invoker"
