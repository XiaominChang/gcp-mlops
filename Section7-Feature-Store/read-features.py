"""
Read Features from Feature Store (Optimized)
==============================================
Demonstrates both online and offline feature retrieval:
  - Online: Use FeatureOnlineStore for low-latency serving
  - Offline: Query BigQuery directly for batch/training data

Requirements:
  pip install google-cloud-aiplatform>=1.60.0 google-cloud-bigquery>=3.20.0

Updated: 2026 - Uses Feature Store (Optimized) API (replaces legacy
         Featurestore.batch_serve_to_df).
"""

from google.cloud import aiplatform
from google.cloud import bigquery

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
BQ_DATASET = "credit_scoring"
BQ_TABLE = "credit_features"
BQ_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

ONLINE_STORE_ID = "credit_scoring_online_store"
FEATURE_VIEW_ID = "credit_features_view"

aiplatform.init(project=PROJECT_ID, location=REGION)
bq_client = bigquery.Client(project=PROJECT_ID)


# ═════════════════════════════════════════════════════════════════════════════
# ONLINE SERVING - Low-latency feature lookups via FeatureOnlineStore
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ONLINE SERVING - FeatureOnlineStore")
print("=" * 60)

# ─── 1. Get the FeatureOnlineStore and FeatureView ────────────────────────────
print("\nStep 1: Connecting to FeatureOnlineStore...")

feature_online_store = aiplatform.FeatureOnlineStore(ONLINE_STORE_ID)
print(f"  Online store: {feature_online_store.resource_name}")

feature_view = feature_online_store.get_feature_view(FEATURE_VIEW_ID)
print(f"  Feature view: {feature_view.resource_name}")

# ─── 2. Fetch features for a single entity (online) ──────────────────────────
print("\nStep 2: Fetching features for a single entity...")

# Fetch features for entity_id = "1"
entity_id = "1"
online_response = feature_view.read(key=[entity_id])

print(f"  Entity ID: {entity_id}")
print(f"  Features:  {online_response.to_dict()}")

# ─── 3. Fetch features for multiple entities (online batch) ──────────────────
print("\nStep 3: Fetching features for multiple entities...")

entity_ids = ["1", "2", "3", "10", "100"]
online_batch_response = feature_view.read(key=entity_ids)

print(f"  Fetched features for {len(entity_ids)} entities")
for resp in online_batch_response:
    print(f"  Entity {resp.entity_id}: credit_amount={resp.features.get('credit_amount')}, "
          f"age={resp.features.get('age')}")


# ═════════════════════════════════════════════════════════════════════════════
# OFFLINE SERVING - Query BigQuery directly for batch/training data
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("OFFLINE SERVING - BigQuery")
print("=" * 60)

# ─── 4. Query all features for specific entities ─────────────────────────────
print("\nStep 4: Querying features from BigQuery (offline)...")

query = f"""
    SELECT
        entity_id,
        credit_amount,
        credit_duration,
        installment_commitment,
        checking_balance,
        savings_balance,
        age,
        credit_standing_code,
        purpose_code,
        feature_timestamp
    FROM `{BQ_TABLE_ID}`
    WHERE entity_id IN ('1', '2', '3', '10', '100')
    ORDER BY entity_id
"""

offline_df = bq_client.query(query).to_dataframe()
print(f"  Retrieved {len(offline_df)} rows from BigQuery")
print(offline_df.to_string(index=False))

# ─── 5. Batch feature retrieval for model training ───────────────────────────
print("\nStep 5: Batch retrieval for model training...")

training_query = f"""
    SELECT
        entity_id,
        credit_amount,
        credit_duration,
        installment_commitment,
        credit_score,
        checking_balance,
        savings_balance,
        existing_credits,
        job_history,
        purpose_code,
        other_parties_code,
        qualification_code,
        assets_code,
        housing_code,
        marital_status_code,
        other_payment_plans_code,
        age,
        num_dependents,
        residence_since,
        sex_code,
        credit_standing_code
    FROM `{BQ_TABLE_ID}`
"""

training_df = bq_client.query(training_query).to_dataframe()
print(f"  Retrieved {len(training_df)} rows for model training")
print(f"  Columns: {list(training_df.columns)}")
print(f"  Shape: {training_df.shape}")

# ─── 6. Point-in-time feature retrieval ──────────────────────────────────────
print("\nStep 6: Point-in-time feature retrieval...")

# Retrieve features as they were at a specific timestamp
# This is critical for training data to avoid data leakage
pit_query = f"""
    SELECT
        entity_id,
        credit_amount,
        age,
        credit_standing_code,
        feature_timestamp
    FROM `{BQ_TABLE_ID}`
    WHERE feature_timestamp <= TIMESTAMP('2026-01-01 00:00:00 UTC')
    ORDER BY entity_id
    LIMIT 10
"""

pit_df = bq_client.query(pit_query).to_dataframe()
print(f"  Point-in-time rows (before 2026-01-01): {len(pit_df)}")
if len(pit_df) > 0:
    print(pit_df.to_string(index=False))

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n=== Feature Retrieval Complete ===")
print("Online serving:  Use feature_view.read() for low-latency lookups")
print("Offline serving: Query BigQuery directly with SQL")
print("Training data:   Use point-in-time queries to avoid data leakage")
