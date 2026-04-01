"""
Create Feature Store (Optimized) on Vertex AI
===============================================
Creates a FeatureOnlineStore and FeatureView using the new
Feature Store (Optimized) API - replaces the legacy Featurestore/EntityType API.

Architecture:
  BigQuery (offline) --> FeatureView (sync) --> FeatureOnlineStore (online serving)

Requirements:
  pip install google-cloud-aiplatform>=1.60.0 google-cloud-bigquery>=3.20.0

Updated: 2026 - Complete rewrite from legacy API to Feature Store (Optimized).
"""

from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud.aiplatform_v1.types import feature_online_store as fos_types

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BQ_DATASET = "credit_scoring"                # BigQuery dataset name
BQ_TABLE = "credit_features"                 # BigQuery table name

aiplatform.init(project=PROJECT_ID, location=REGION)

# ─── 1. Create a BigQuery dataset and table for features ─────────────────────
print("Step 1: Setting up BigQuery table for feature data...")

bq_client = bigquery.Client(project=PROJECT_ID)

# Create dataset if it doesn't exist
dataset_ref = bigquery.DatasetReference(PROJECT_ID, BQ_DATASET)
try:
    bq_client.get_dataset(dataset_ref)
    print(f"  Dataset {BQ_DATASET} already exists.")
except Exception:
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = REGION
    bq_client.create_dataset(dataset)
    print(f"  Created dataset: {BQ_DATASET}")

# Define the schema for the credit scoring features table
schema = [
    bigquery.SchemaField("entity_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED"),
    # Credit request features
    bigquery.SchemaField("credit_amount", "FLOAT64"),
    bigquery.SchemaField("credit_duration", "FLOAT64"),
    bigquery.SchemaField("installment_commitment", "FLOAT64"),
    bigquery.SchemaField("credit_score", "FLOAT64"),
    # Customer financial profile features
    bigquery.SchemaField("checking_balance", "FLOAT64"),
    bigquery.SchemaField("savings_balance", "FLOAT64"),
    bigquery.SchemaField("existing_credits", "FLOAT64"),
    bigquery.SchemaField("job_history", "FLOAT64"),
    # Credit context features
    bigquery.SchemaField("purpose_code", "FLOAT64"),
    bigquery.SchemaField("other_parties_code", "FLOAT64"),
    bigquery.SchemaField("qualification_code", "FLOAT64"),
    bigquery.SchemaField("credit_standing_code", "FLOAT64"),
    bigquery.SchemaField("assets_code", "FLOAT64"),
    bigquery.SchemaField("housing_code", "FLOAT64"),
    bigquery.SchemaField("marital_status_code", "FLOAT64"),
    bigquery.SchemaField("other_payment_plans_code", "FLOAT64"),
    # Customer demographics features
    bigquery.SchemaField("age", "FLOAT64"),
    bigquery.SchemaField("num_dependents", "FLOAT64"),
    bigquery.SchemaField("residence_since", "FLOAT64"),
    bigquery.SchemaField("sex_code", "FLOAT64"),
]

table_ref = dataset_ref.table(BQ_TABLE)
table = bigquery.Table(table_ref, schema=schema)
try:
    bq_client.get_table(table_ref)
    print(f"  Table {BQ_TABLE} already exists.")
except Exception:
    bq_client.create_table(table)
    print(f"  Created table: {BQ_DATASET}.{BQ_TABLE}")

BQ_URI = f"bq://{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
print(f"  BigQuery URI: {BQ_URI}")

# ─── 2. Create FeatureOnlineStore ────────────────────────────────────────────
print("\nStep 2: Creating FeatureOnlineStore...")

# The FeatureOnlineStore is the top-level resource for online serving.
# It manages a Bigtable instance for low-latency feature lookups.

ONLINE_STORE_ID = "credit_scoring_online_store"

feature_online_store = aiplatform.FeatureOnlineStore.create_bigtable_store(
    name=ONLINE_STORE_ID,
    labels={"domain": "credit-scoring", "env": "dev"},
)

print(f"  FeatureOnlineStore created: {feature_online_store.resource_name}")

# ─── 3. Create FeatureView ──────────────────────────────────────────────────
print("\nStep 3: Creating FeatureView...")

# The FeatureView links the BigQuery source to the online store.
# It defines which columns to sync and the sync schedule.

FEATURE_VIEW_ID = "credit_features_view"

feature_view = feature_online_store.create_feature_view(
    name=FEATURE_VIEW_ID,
    source=aiplatform.FeatureView.BigQuerySource(
        uri=BQ_URI,
        entity_id_columns=["entity_id"],
    ),
    # Sync schedule: every 24 hours (cron expression)
    sync_config=aiplatform.FeatureView.SyncConfig(cron="0 0 * * *"),
)

print(f"  FeatureView created: {feature_view.resource_name}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Feature Store Setup Complete ===")
print(f"  BigQuery source:       {BQ_URI}")
print(f"  FeatureOnlineStore:    {feature_online_store.resource_name}")
print(f"  FeatureView:           {feature_view.resource_name}")
print(f"  Sync schedule:         Daily at midnight UTC")
print("\nNext steps:")
print("  1. Ingest data into the BigQuery table (see ingest-features.py)")
print("  2. Trigger a sync or wait for the cron schedule")
print("  3. Read features online or query BigQuery for offline access")
