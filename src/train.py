"""
Train a churn-prediction model and log the run with MLflow.

Usage:
    python -m src.train --data data/telco_churn.csv --model-out models/model.joblib
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocess import build_preprocessor, load_raw, split_features_target

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
    ),
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco_churn.csv")
    parser.add_argument("--model-out", default="models/model.joblib")
    parser.add_argument("--metrics-out", default="models/metrics.json")
    parser.add_argument("--experiment", default="telco-churn")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)

    df = load_raw(args.data)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_run = None
    best_f1 = -1.0

    for model_name, estimator in CANDIDATE_MODELS.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(steps=[
                ("preprocess", build_preprocessor()),
                ("classifier", estimator),
            ])

            start = time.time()
            pipeline.fit(X_train, y_train)
            train_seconds = time.time() - start

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_metric("train_seconds", train_seconds)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(
                pipeline, artifact_path="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )

            print(f"[{model_name}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_run = (model_name, pipeline, metrics)

    model_name, pipeline, metrics = best_run
    print(f"\nBest model: {model_name} (f1={best_f1:.4f}) -> saving to {args.model_out}")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump({"best_model": model_name, **metrics}, f, indent=2)


if __name__ == "__main__":
    main()
