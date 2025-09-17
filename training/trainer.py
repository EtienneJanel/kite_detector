import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from training.common.mlflow_utils import compute_detection_metrics, log_model_params
from training.model import KiteDetector, SpotDetector

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()
IMAGES_DIR = CONFIG["IMAGES_DIR"]


def load_dataset(csv_path):
    """
    dataset for Spot Trainer - csv file with columns: [image_path,label]
    Returns
        X: image_path
        y: label
    """

    df = pd.read_csv(csv_path)
    assert "image_path" in df.columns and "label" in df.columns
    return df["image_path"].to_numpy(), df["label"].to_numpy()


def spot_trainer(dataset_path, experiment_name, run_name, model_version):
    model_name = f"{experiment_name.replace(' ', '_')}_{model_version}.pkl"
    model_path = Path("training/models") / model_name
    if model_path.exists():
        raise AssertionError(f"model already exists {model_path}")
    # Load and split arrays into random train and test subsets.
    X, y = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_size", len(X))
        mlflow.log_param("X_train_size", len(X_train))
        mlflow.log_param("X_test_size", len(X_test))

        # Init and train model
        model = SpotDetector()
        log_model_params(model)
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        logger.info(f"Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)

        for cls, metrics in report.items():
            if isinstance(metrics, dict):  # skip 'accuracy' key
                for k, v in metrics.items():
                    mlflow.log_metric(f"{cls}_{k}", v)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")


def kite_trainer(dataset_path, experiment_name, run_name, model_version):
    training_log_path = (
        Path("training/models") / f"{experiment_name.replace(' ', '_')}_{model_version}"
    )
    with open(dataset_path, "r") as f:
        coco = json.load(f)

    # Map image_id to file path
    image_id_to_path = {
        img["id"]: img["path"].replace("datasets", IMAGES_DIR) for img in coco["images"]
    }

    # Group annotations per image
    image_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        image_id_to_anns[ann["image_id"]].append(ann)

    image_paths = []
    annotations = []

    for image_id, path in image_id_to_path.items():
        # overwritting category 1 to 38, corresponding to "kite" in yolo
        full_path = Path(path)
        image_paths.append(full_path)
        for bbox_item in image_id_to_anns[image_id]:
            bbox_item["category_id"] = 38
        
        annotations.append(image_id_to_anns[image_id])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, annotations, test_size=0.2, random_state=42
    )

    # MLflow setup
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_size", len(image_paths))
        mlflow.log_param("X_train_size", len(X_train))
        mlflow.log_param("X_test_size", len(X_test))

        # Init and train
        model = KiteDetector(training_log_path=training_log_path)
        log_model_params(model)
        model.fit(X_train, y_train)

        # Predict and evaluate
        preds = model.predict(X_test)

        # Basic detection metrics (true positives, precision, recall)
        metrics = compute_detection_metrics(preds, y_test, iou_threshold=0.5)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            logger.info(f"{key}: {value:.4f}")

        model.save_model()
        logger.info(f"Model saved in {training_log_path}")


if __name__ == "__main__":
    model_options = {
        "spot": spot_trainer,
        "kite": kite_trainer,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Which model to train: `spot` for scene detection, `kite` for object detection",
        choices=["spot", "kite"],
    )

    parser.add_argument(
        "-d", "--dataset_path", type=str, help="Path to CSV with image_path, label"
    )
    parser.add_argument(
        "-e", "--experiment_name", type=str, help="Experiment name in MLFlow"
    )
    parser.add_argument(
        "-r",
        "--run_name",
        type=str,
        help="Run name in MLFlow, belongs to an experiment",
    )
    parser.add_argument("-v", "--model_version", type=str, help="version of the model")

    args = parser.parse_args()
    model_options.get(args.model)(
        dataset_path=args.dataset_path,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model_version=args.model_version,
    )
