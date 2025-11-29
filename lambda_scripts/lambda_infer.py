import json
import os
import boto3

runtime = boto3.client("sagemaker-runtime")
ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]

def lambda_handler(event, context):
    # event["body"] will have the JSON from your website
    raw_body = event.get("body")

    # If body is a JSON string, parse it
    if isinstance(raw_body, dict):
        payload = raw_body
    elif isinstance(raw_body, str):
        payload = json.loads(raw_body)
    elif isinstance(raw_body, list):
        payload = json.loads(raw_body or "{}")

    # Call SageMaker endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = response["Body"].read().decode("utf-8")
    
    # Log the result
    print(result)

    # HTTP response back to frontend (CORS open for now)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": result,
    }
