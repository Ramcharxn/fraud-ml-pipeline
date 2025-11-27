# fraud_sagemaker.py
#
# SageMaker Script Mode entry point
# - When run as a training job: trains LightGBM pipeline and saves it
# - When used for inference: model_fn() is called to load the pipeline

import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import lightgbm as lgb


# ---------- Simple feature engineering helpers ----------

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal feature engineering similar to what we did in the notebook.
    You can extend this as needed to match your final notebook pipeline.
    """
    # TransactionDT-derived features
    if "TransactionDT" in df.columns:
        df["DT_day"] = df["TransactionDT"] // (24 * 60 * 60)
        df["DT_hour"] = (df["TransactionDT"] // 3600) % 24
        df["DT_week"] = df["DT_day"] // 7
        df["DT_month"] = df["DT_day"] // 30

    # Log amount
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # Handle dist1/dist2
    if "dist1" in df.columns:
        df["dist1"] = df["dist1"].fillna(-1)
    if "dist2" in df.columns:
        df = df.drop(columns=["dist2"])

    # Simple frequency encodings for a few key features (if present)
    freq_cols = ["card1", "card2", "addr1", "addr2"]
    for col in freq_cols:
        if col in df.columns:
            freq = df[col].value_counts()
            df[col + "_freq"] = df[col].map(freq)

    return df


# ---------- Training entry point ----------

def train(args):
    """
    Main training loop for SageMaker.
    Reads training data from SM_CHANNEL_TRAINING, trains pipeline, saves model.
    """
    training_dir = args.train  # passed by SageMaker as SM_CHANNEL_TRAINING
    model_dir = args.model_dir

    # Expecting these CSVs inside the 'training' channel
    train_path = os.path.join(training_dir, 'train_merged1.csv')

    print("Reading training data...")
    df = pd.read_csv(train_path)

    print("Raw train shape:", df.shape)

    # Minimal feature engineering
    df = basic_feature_engineering(df)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Target
    if "isFraud" not in df.columns:
        raise ValueError("Expected 'isFraud' column in training data.")

    y = df["isFraud"].astype(int)
    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    # Fill missing values (numeric vs non-numeric)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    X[num_cols] = X[num_cols].fillna(-1)
    X[cat_cols] = X[cat_cols].fillna("missing")

    print("After basic preprocessing, X shape:", X.shape)

    # Train/valid split for quick evaluation (not used by SageMaker beyond logs)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ColumnTransformer: numeric passthrough, categorical ordinal encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
        ]
    )

    # LightGBM classifier (params can be tuned)
    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)

    # Simple metrics for logs
    valid_pred = pipeline.predict_proba(X_valid)[:, 1]
    roc = roc_auc_score(y_valid, valid_pred)
    precision, recall, _ = precision_recall_curve(y_valid, valid_pred)
    pr_auc = auc(recall, precision)

    print(f"Validation ROC-AUC: {roc:.5f}")
    print(f"Validation PR-AUC : {pr_auc:.5f}")

    # Save model to model_dir (SageMaker will tar/zip this for you)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)

    print(f"Saved model to {model_path}")


# ---------- Inference hooks for SageMaker SKLearn container ----------

def model_fn(model_dir):
    """Load pipeline from the model_dir path."""
    model_path = os.path.join(model_dir, "model.joblib")
    print(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline


def input_fn(request_body, content_type="application/json"):
    """Parse incoming request payload into a pandas DataFrame."""
    import json

    if content_type == "application/json":
        data = json.loads(request_body)
        # Expect either a dict of lists or a list of dicts
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported JSON format for input.")
        return df

    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Run prediction.
    input_data: pandas DataFrame (decoded from JSON by input_fn)
    model: loaded sklearn Pipeline
    """

    # 1) Copy to avoid modifying original
    df = input_data.copy()

    # 2) Apply the SAME feature engineering as in training
    df = basic_feature_engineering(df)

    # 3) Drop columns we dropped in training
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 4) Get the feature lists that the preprocessor expects
    preprocessor = model.named_steps["preprocess"]
    num_cols = list(preprocessor.transformers_[0][2])  # numeric_features
    cat_cols = list(preprocessor.transformers_[1][2])  # categorical_features

    # 5) Ensure all expected columns exist
    #    - For missing numeric columns, fill with -1
    #    - For missing categorical columns, fill with "missing"
    for col in num_cols:
        if col not in df.columns:
            df[col] = -1
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "missing"

    # 6) Fill NaNs consistently with training logic
    df[num_cols] = df[num_cols].fillna(-1)
    df[cat_cols] = df[cat_cols].fillna("missing")

    # 7) Extra columns are fine — ColumnTransformer will just select num_cols + cat_cols
    #    Run the full pipeline
    preds = model.predict_proba(df)[:, 1]

    return preds



def output_fn(prediction, accept="application/json"):
    """Format predictions into response."""
    import json

    if accept == "application/json":
        return json.dumps({"predictions": prediction.tolist()}), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# ---------- Main entry point ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters (can be overridden from SageMaker)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)

    # SageMaker-specific arguments (set by the training environment)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )

    args = parser.parse_args()

    train(args)
