import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from loguru import logger

from training.common.mlflow_utils import compute_detection_metrics
from training.model import KiteDetector

from config.config_loader import load_config

load_dotenv()

CONFIG = load_config()
IMAGES_DIR = CONFIG["IMAGES_DIR"]


def load_kite_data(coco_json_path):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    image_id_to_path = {
        img["id"]: img["path"].replace("datasets", IMAGES_DIR) for img in coco["images"]
    }

    image_id_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        image_id_to_anns[ann["image_id"]].append(ann)

    image_paths = []
    annotations = []

    for image_id, path in image_id_to_path.items():
        full_path = Path(path)
        image_paths.append(full_path)

        for bbox_item in image_id_to_anns[image_id]:
            bbox_item["category_id"] = 38  # Map to 'kite' class
        annotations.append(image_id_to_anns[image_id])

    return image_paths, annotations


def benchmark_kite_model(dataset_path, experiment_name, run_name, model_ckpt=None, fine_tuned_model_path=None):
    # Load test dataset
    image_paths, annotations = load_kite_data(dataset_path)

    # Use full dataset for evaluation
    X_test = image_paths
    y_test = annotations

    # MLflow setup
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_size", len(image_paths))
        mlflow.log_param("X_test_size", len(X_test))
        mlflow.log_param("benchmark_mode", "fine-tuned" if fine_tuned_model_path else "pretrained")
        mlflow.log_param("model_ckpt", model_ckpt)

        if fine_tuned_model_path:
            logger.info(f"Loading fine-tuned model from {fine_tuned_model_path}")
            model = KiteDetector.load_model(fine_tuned_model_path)
        else:
            logger.info(f"Loading pretrained model {model_ckpt}")
            model = KiteDetector(model_ckpt=model_ckpt)

        # Evaluate
        preds = model.predict(X_test)
        metrics = compute_detection_metrics(preds, y_test, iou_threshold=0.5)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            logger.info(f"{key}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset_path", type=str, required=True, help="Path to COCO JSON dataset"
    )
    parser.add_argument(
        "-e", "--experiment_name", type=str, required=True, help="Experiment name in MLFlow"
    )
    parser.add_argument(
        "-r", "--run_name", type=str, required=True, help="Run name in MLFlow"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        help="HuggingFace model checkpoint to benchmark, e.g. 'hustvl/yolos-tiny'. Leave empty if using fine-tuned model.",
    )
    parser.add_argument(
        "--fine_tuned_model_path",
        type=str,
        help="Path to fine-tuned model directory (e.g. training/models/kite_v1). Leave empty to benchmark original HuggingFace model.",
    )

    args = parser.parse_args()

    benchmark_kite_model(
        dataset_path=args.dataset_path,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model_ckpt=args.model_ckpt,
        fine_tuned_model_path=args.fine_tuned_model_path,
    )
