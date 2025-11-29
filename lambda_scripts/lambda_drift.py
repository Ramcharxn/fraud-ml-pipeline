import os
import json
import boto3
import csv
from io import StringIO

s3 = boto3.client("s3")
codepipeline = boto3.client("codepipeline")

BUCKET = os.environ["BUCKET_NAME"]
TRAINING_KEY = os.environ["TRAINING_KEY"]        # e.g. raw/train_merged1.csv
CAPTURE_PREFIX = os.environ["CAPTURE_PREFIX"]    # e.g. data-capture/
PIPELINE_NAME = os.environ["PIPELINE_NAME"]      # e.g. fraud-ml-pipeline

# Simple helper: compute mean of a numeric column from CSV rows
def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0

def load_training_stats(max_rows=50000):
    obj = s3.get_object(Bucket=BUCKET, Key=TRAINING_KEY)
    body = obj["Body"]

    # Stream lines instead of loading whole file into memory
    lines = (line.decode("utf-8") for line in body.iter_lines())

    reader = csv.DictReader(lines)

    amounts = []
    for i, row in enumerate(reader):
        if i >= max_rows:   # only sample first N rows
            break
        try:
            amt = float(row.get("TransactionAmt", "0") or 0)
            amounts.append(amt)
        except ValueError:
            continue

    return {"mean_amt": mean(amounts)}


def get_latest_capture_object():
    # list all objects under capture prefix, pick the newest
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=CAPTURE_PREFIX)
    contents = resp.get("Contents", [])
    if not contents:
        return None

    latest = max(contents, key=lambda x: x["LastModified"])
    return latest["Key"]

def load_capture_stats(latest_key):
    obj = s3.get_object(Bucket=BUCKET, Key=latest_key)
    body = obj["Body"].read().decode("utf-8")

    amounts = []

    print('this is a lambda output of body',body)

    # SageMaker data capture is JSON lines
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        print('this is a lambda output of rec', rec)
        # We expect the request payload to be JSON with TransactionAmt
        # This depends on how you send features; adjust if needed
        capture = rec.get("captureData", {}) or {}
        endpoint_input = capture.get("endpointInput", {}) or {}

        print("this is a lambda output of endpoint_input", endpoint_input)

        data_str = endpoint_input.get("data") or endpoint_input.get("body") or ""

        print('this is a lambda output of data_str', data_str)

        # If body is JSON string, parse it
        try:
            payload = json.loads(data_str)
        except Exception:
            continue

        vals = payload.get("TransactionAmt", [])
        if isinstance(vals, list):
            for v in vals:
                try:
                    amounts.append(float(v))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                amounts.append(float(vals))
            except (TypeError, ValueError):
                pass

    return {"mean_amt": mean(amounts)}

def should_trigger_drift_retrain(train_stats, capture_stats):
    # Relative change in mean TransactionAmt
    base = train_stats["mean_amt"]
    live = capture_stats["mean_amt"]
    if base == 0:
        return False

    try:
        threshold = float(os.environ.get("DRIFT_THRESHOLD", "0.3"))
    except ValueError:
        threshold = 0.3

    rel_change = abs(live - base) / base
    return rel_change >= threshold

def start_pipeline():
    response = codepipeline.start_pipeline_execution(
        name=PIPELINE_NAME
    )
    return response["pipelineExecutionId"]

def lambda_handler(event, context):
    # 1) Load baseline stats
    train_stats = load_training_stats()

    # 2) Get latest captured batch
    latest_key = get_latest_capture_object()
    if not latest_key:
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "No capture data yet, skipping"})
        }

    capture_stats = load_capture_stats(latest_key)

    # 3) Decide drift
    drift = should_trigger_drift_retrain(train_stats, capture_stats)

    if drift:
        exec_id = start_pipeline()
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Drift detected, retraining triggered",
                "pipelineExecutionId": exec_id,
                "train_stats": train_stats,
                "capture_stats": capture_stats
            })
        }
    else:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "No significant drift, no retrain",
                "train_stats": train_stats,
                "capture_stats": capture_stats
            })
        }
