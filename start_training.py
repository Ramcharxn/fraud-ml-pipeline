# start_training.py
#
# Run inside CodeBuild.
# Uses SageMaker SKLearn estimator with fraud_sagemaker.py as entry point:
#  - builds a combined training dataset (historical + recent captured data)
#  - uploads it to S3
#  - launches a SageMaker training job
#  - deploys/updates the existing endpoint with the new model

import os
import json
from datetime import datetime, timedelta
from io import StringIO

import boto3
import pandas as pd
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.predictor import Predictor
from sagemaker.model_monitor import DataCaptureConfig
from botocore.exceptions import ClientError
import boto3

# --------------------------------------------------------------------
# Environment / configuration
# --------------------------------------------------------------------

region = os.environ.get("AWS_REGION", "us-east-1")
role_arn = os.environ["SAGEMAKER_ROLE_ARN"]
bucket = os.environ["S3_BUCKET"]
endpoint_name = os.environ["ENDPOINT_NAME"]
sm_client = boto3.client("sagemaker", region_name=region)
s3 = boto3.client("s3", region_name=region)

# Base training data key (historical labeled data)
BASE_TRAIN_KEY = os.environ.get("BASE_TRAIN_KEY", "raw/train_merged1.csv")

# Where to write the dynamically combined training data
COMBINED_TRAIN_KEY = os.environ.get("COMBINED_TRAIN_KEY", "raw/train_merged_with_capture.csv")

# Data-capture prefix and recency window (for new/live data)
CAPTURE_PREFIX = os.environ.get("CAPTURE_PREFIX", "data-capture/")
RECENT_DAYS = int(os.environ.get("RECENT_DAYS", "30"))

data_capture_s3 = f"s3://{bucket}/{CAPTURE_PREFIX}"

# --------------------------------------------------------------------
# Helpers to load data from S3
# --------------------------------------------------------------------


def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read a CSV from S3 into a DataFrame using boto3."""
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(body))


def load_base_training_data() -> pd.DataFrame:
    print(f"Loading base training data from s3://{bucket}/{BASE_TRAIN_KEY}")
    df = read_csv_from_s3(bucket, BASE_TRAIN_KEY)
    print("Base training data shape:", df.shape)
    return df


def parse_capture_record(line: str):
    line = line.strip()
    if not line:
        return None, None

    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        return None, None

    capture = rec.get("captureData", {})
    endpoint_input = capture.get("endpointInput", {})
    endpoint_output = capture.get("endpointOutput", {})

    # Input payload: the original request body as a JSON string
    data_str = endpoint_input.get("data")
    if not data_str:
        return None, None

    try:
        payload = json.loads(data_str)
    except Exception:
        return None, None

    # Flatten: most of your features are lists with a single element -> take index 0
    features = {}
    for k, v in payload.items():
        if isinstance(v, list) and v:
            features[k] = v[0]
        else:
            features[k] = v

    # Output payload: model prediction(s)
    pred_prob = None
    out_str = endpoint_output.get("data")
    if out_str:
        try:
            out_payload = json.loads(out_str)
            preds = out_payload.get("predictions")
            if isinstance(preds, list) and preds:
                pred_prob = float(preds[0])
        except Exception:
            pred_prob = None

    return features, pred_prob


def load_recent_captured_data(bucket: str, prefix: str, recent_days: int) -> pd.DataFrame:
    cutoff = datetime.utcnow() - timedelta(days=recent_days)
    print(f"Loading captured data from s3://{bucket}/{prefix} newer than {cutoff.isoformat()}")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    rows = []

    for page in pages:
        for obj in page.get("Contents", []):
            if obj["LastModified"] < cutoff:
                continue

            key = obj["Key"]
            print(f"Reading capture file: {key}")
            file_obj = s3.get_object(Bucket=bucket, Key=key)
            body = file_obj["Body"].read().decode("utf-8")

            for line in body.splitlines():
                features, pred_prob = parse_capture_record(line)
                if features is None:
                    continue

                # Add prediction info
                row = dict(features)
                if pred_prob is not None:
                    row["prediction_prob"] = pred_prob
                    # Pseudo-label: classify >= 0.5 as fraud (1), else 0
                    row["isFraud"] = int(pred_prob >= 0.5)

                rows.append(row)

    if not rows:
        print("No recent captured records found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print("Recent captured data shape:", df.shape)
    return df


def build_combined_training_data() -> str:
    """Return the S3 URI to the combined training CSV (historical + recent)."""
    base_df = load_base_training_data()
    recent_df = load_recent_captured_data(bucket, CAPTURE_PREFIX, RECENT_DAYS)

    if recent_df.empty:
        print("No recent captured data; using base training data only.")
        combined_df = base_df
    else:
        # Make sure recent_df has at least the same columns as base_df.
        for col in base_df.columns:
            if col not in recent_df.columns:
                recent_df[col] = pd.NA

        # Re-order columns to match base_df
        recent_df = recent_df[base_df.columns]

        combined_df = pd.concat([base_df, recent_df], ignore_index=True)
        print("Combined training data shape:", combined_df.shape)

    # Write to /tmp and upload to S3
    local_path = "/tmp/train_combined.csv"
    combined_df.to_csv(local_path, index=False)

    print(f"Uploading combined training data to s3://{bucket}/{COMBINED_TRAIN_KEY}")
    s3.upload_file(local_path, bucket, COMBINED_TRAIN_KEY)

    return f"s3://{bucket}/{COMBINED_TRAIN_KEY}"


def main():
    train_data_uri = build_combined_training_data()
    sagemaker_session = sagemaker.Session()

    estimator = SKLearn(
        entry_point="fraud_sagemaker.py",
        role=role_arn,
        instance_type="ml.m5.xlarge",
        framework_version="1.2-1",
        py_version="py3",
        base_job_name="fraud-lgbm-train",
        sagemaker_session=sagemaker_session,
        hyperparameters={
            "num-leaves": 64,
            "n-estimators": 800,
            "learning-rate": 0.03,
        },
    )

    # Training input: point to the S3 prefix with your CSVs
    train_input = sagemaker.inputs.TrainingInput(
        s3_data=train_data_uri,
        content_type="text/csv"
    )

    print("Starting training job...")
    estimator.fit({"training": train_input})
    print("Training job completed.")


    def endpoint_exists(name: str) -> bool:
        try:
            sm_client.describe_endpoint(EndpointName=name)
            return True
        except ClientError:
            return False


    # 1. Delete endpoint (and its config) if it exists
    if endpoint_exists(endpoint_name):
        print("Deleting existing endpoint:", endpoint_name)
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        try:
            sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        except ClientError as e:
            # Config might already be gone; ignore that case
            print("Warning while deleting old endpoint config:", e)

        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
        print("Endpoint deleted:", endpoint_name)

    # 2. Configure data capture for the new endpoint
    data_capture_config = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=data_capture_s3,
        csv_content_types=["text/csv"],
        json_content_types=["application/json"],
    )

    # 3. Deploy fresh endpoint with data capture enabled
    estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name,
        data_capture_config=data_capture_config,
    )

    print("Endpoint created with data capture enabled:", endpoint_name)


if __name__ == "__main__":
    main()
