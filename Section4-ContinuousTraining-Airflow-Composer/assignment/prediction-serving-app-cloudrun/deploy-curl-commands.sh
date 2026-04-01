#!/bin/bash

# Build and push to Artifact Registry
docker build -t roi_model_serving .

docker tag roi_model_serving us-docker.pkg.dev/udemy-mlops/mlops-repo/roi_model_serving:latest

docker push us-docker.pkg.dev/udemy-mlops/mlops-repo/roi_model_serving:latest

# Deploy to Cloud Run
gcloud run deploy roi-model-inference \
  --image us-docker.pkg.dev/udemy-mlops/mlops-repo/roi_model_serving:latest \
  --region us-central1

# Test locally
curl -X POST http://127.0.0.1:5050/predict \
  -H "Content-Type: application/json" \
  -d '{"EMAIL":521698,"SEARCH_ENGINE":521339,"SOCIAL_MEDIA":521528,"VIDEO":519625}'

# Test on Cloud Run (update URL after deployment)
# curl -X POST https://roi-model-inference-XXXXX-uc.a.run.app/predict \
#   -H "Content-Type: application/json" \
#   -d '{"EMAIL":520742,"SEARCH_ENGINE":522807,"SOCIAL_MEDIA":518987,"VIDEO":521946}'
