"""
Ingest Features into BigQuery for Feature Store (Optimized)
============================================================
Reads the credit scoring CSV, preprocesses it, and loads into
the BigQuery table that backs the Feature Store FeatureView.

In the Optimized Feature Store, BigQuery IS the offline store.
Data ingested into BigQuery is automatically synced to the
FeatureOnlineStore on the configured schedule.

Requirements:
  pip install google-cloud-bigquery>=3.20.0 pandas>=2.2.0
            google-cloud-storage

Updated: 2026 - Uses BigQuery as native feature source (replaces
         legacy EntityType.ingest_from_df).
"""

import pandas as pd
from google.cloud import bigquery, storage
from datetime import datetime, timezone

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ID = "your-gcp-project-id"          # TODO: replace
REGION = "us-central1"
GCS_BUCKET = "your-bucket-name"              # TODO: replace
BQ_DATASET = "credit_scoring"
BQ_TABLE = "credit_features"
BQ_TABLE_ID = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

bq_client = bigquery.Client(project=PROJECT_ID)


# ─── Encoding Functions (same as preprocessing) ─────────────────────────────
def purpose_encode(x):
    mapping = {"Consumer Goods": 1, "Vehicle": 2, "Tuition": 3, "Business": 4, "Repairs": 5}
    return mapping.get(x, 0)


def other_parties_encode(x):
    mapping = {"Guarantor": 1, "Co-Applicant": 2}
    return mapping.get(x, 0)


def qualification_encode(x):
    mapping = {"unskilled": 1, "skilled": 2, "highly skilled": 3}
    return mapping.get(x, 0)


def credit_standing_encode(x):
    return 1 if x == "good" else 0


def assets_encode(x):
    mapping = {"Vehicle": 1, "Investments": 2, "Home": 3}
    return mapping.get(x, 0)


def housing_encode(x):
    mapping = {"rent": 1, "own": 2}
    return mapping.get(x, 0)


def marital_status_encode(x):
    mapping = {"Married": 1, "Single": 2}
    return mapping.get(x, 0)


def other_payment_plans_encode(x):
    mapping = {"bank": 1, "stores": 2}
    return mapping.get(x, 0)


def sex_encode(x):
    return 1 if x == "M" else 0


def preprocess_data(df):
    """Apply categorical encoding to all text columns."""
    df["PURPOSE_CODE"] = df["PURPOSE"].apply(purpose_encode)
    df["OTHER_PARTIES_CODE"] = df["OTHER_PARTIES"].apply(other_parties_encode)
    df["QUALIFICATION_CODE"] = df["QUALIFICATION"].apply(qualification_encode)
    df["CREDIT_STANDING_CODE"] = df["CREDIT_STANDING"].apply(credit_standing_encode)
    df["ASSETS_CODE"] = df["ASSETS"].apply(assets_encode)
    df["HOUSING_CODE"] = df["HOUSING"].apply(housing_encode)
    df["MARITAL_STATUS_CODE"] = df["MARITAL_STATUS"].apply(marital_status_encode)
    df["OTHER_PAYMENT_PLANS_CODE"] = df["OTHER_PAYMENT_PLANS"].apply(other_payment_plans_encode)
    df["SEX_CODE"] = df["SEX"].apply(sex_encode)

    columns_to_drop = [
        "PURPOSE", "OTHER_PARTIES", "QUALIFICATION", "CREDIT_STANDING",
        "ASSETS", "HOUSING", "MARITAL_STATUS", "OTHER_PAYMENT_PLANS", "SEX",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


# ─── 1. Load and preprocess data ─────────────────────────────────────────────
print("Step 1: Loading and preprocessing credit scoring data...")

input_file = f"gs://{GCS_BUCKET}/credit-scoring/credit_files.csv"
df = pd.read_csv(input_file)
credit_df = preprocess_data(df)

print(f"  Loaded {len(credit_df)} records with {len(credit_df.columns)} columns")

# ─── 2. Prepare DataFrame for BigQuery ───────────────────────────────────────
print("Step 2: Preparing data for BigQuery ingestion...")

# Add entity_id (using CREDIT_REQUEST_ID) and feature_timestamp
current_time = datetime.now(timezone.utc)

bq_df = pd.DataFrame({
    "entity_id": credit_df["CREDIT_REQUEST_ID"].astype(str),
    "feature_timestamp": current_time,
    # Credit request features
    "credit_amount": credit_df["CREDIT_AMOUNT"].astype(float),
    "credit_duration": credit_df["CREDIT_DURATION"].astype(float),
    "installment_commitment": credit_df["INSTALLMENT_COMMITMENT"].astype(float),
    "credit_score": credit_df["CREDIT_SCORE"].astype(float),
    # Customer financial profile features
    "checking_balance": credit_df["CHECKING_BALANCE"].astype(float),
    "savings_balance": credit_df["SAVINGS_BALANCE"].astype(float),
    "existing_credits": credit_df["EXISTING_CREDITS"].astype(float),
    "job_history": credit_df["JOB_HISTORY"].astype(float),
    # Credit context features
    "purpose_code": credit_df["PURPOSE_CODE"].astype(float),
    "other_parties_code": credit_df["OTHER_PARTIES_CODE"].astype(float),
    "qualification_code": credit_df["QUALIFICATION_CODE"].astype(float),
    "credit_standing_code": credit_df["CREDIT_STANDING_CODE"].astype(float),
    "assets_code": credit_df["ASSETS_CODE"].astype(float),
    "housing_code": credit_df["HOUSING_CODE"].astype(float),
    "marital_status_code": credit_df["MARITAL_STATUS_CODE"].astype(float),
    "other_payment_plans_code": credit_df["OTHER_PAYMENT_PLANS_CODE"].astype(float),
    # Customer demographics features
    "age": credit_df["AGE"].astype(float),
    "num_dependents": credit_df["NUM_DEPENDENTS"].astype(float),
    "residence_since": credit_df["RESIDENCE_SINCE"].astype(float),
    "sex_code": credit_df["SEX_CODE"].astype(float),
})

print(f"  Prepared {len(bq_df)} rows for BigQuery")
print(f"  Columns: {list(bq_df.columns)}")

# ─── 3. Load data into BigQuery ──────────────────────────────────────────────
print("Step 3: Loading data into BigQuery...")

job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
)

load_job = bq_client.load_table_from_dataframe(
    bq_df, BQ_TABLE_ID, job_config=job_config
)
load_job.result()  # Wait for completion

table = bq_client.get_table(BQ_TABLE_ID)
print(f"  Loaded {table.num_rows} rows into {BQ_TABLE_ID}")

# ─── 4. Verify data in BigQuery ──────────────────────────────────────────────
print("\nStep 4: Verifying data in BigQuery...")

query = f"""
    SELECT entity_id, credit_amount, credit_duration, age, credit_standing_code
    FROM `{BQ_TABLE_ID}`
    LIMIT 5
"""
result = bq_client.query(query).to_dataframe()
print("  Sample rows:")
print(result.to_string(index=False))

# ─── 5. Trigger FeatureView sync (optional) ──────────────────────────────────
print("\nStep 5: To trigger an immediate sync to the online store, run:")
print("  feature_view.sync()")
print("  Or wait for the scheduled cron sync (daily at midnight UTC)")

print("\n=== Feature Ingestion Complete ===")
print(f"  BigQuery table: {BQ_TABLE_ID}")
print(f"  Rows ingested:  {table.num_rows}")
print(f"  Timestamp:       {current_time.isoformat()}")
