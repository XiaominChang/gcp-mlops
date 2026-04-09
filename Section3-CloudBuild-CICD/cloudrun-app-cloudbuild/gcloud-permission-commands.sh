# Assign Service account user role to the service account 
gcloud projects add-iam-policy-binding udemy-mlops-492103 \
 --member=serviceAccount:188673622020@cloudbuild.gserviceaccount.com --role=roles/iam.serviceAccountUser
 # Assign Cloud Run role to the service account 
gcloud projects add-iam-policy-binding udemy-mlops-492103 \
  --member=serviceAccount:188673622020@cloudbuild.gserviceaccount.com --role=roles/run.admin
gcloud projects add-iam-policy-binding udemy-mlops-492103 \
  --member=serviceAccount:188673622020@cloudbuild.gserviceaccount.com \
  --role=roles/artifactregistry.writer

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
