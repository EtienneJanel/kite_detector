import mlflow
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def log_sklearn_estimator_params(estimator, prefix):
    """
    Logs all hyperparameters of an sklearn estimator or transformer.
    """
    if isinstance(estimator, BaseEstimator):
        params = estimator.get_params()
        for k, v in params.items():
            # Avoid logging overly long or unserializable params
            try:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    mlflow.log_param(f"{prefix}__{k}", v)
                else:
                    mlflow.log_param(f"{prefix}__{k}", str(type(v)))
            except Exception as e:
                print(f"[Warning] Could not log param {k}: {e}")


def log_model_params(model, prefix="model"):
    """
    Automatically logs all parameters from a custom model class
    and its internal sklearn pipeline/model into MLflow.
    """
    # Log attributes from the top-level custom model
    if hasattr(model, "__dict__"):
        for k, v in vars(model).items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(f"{prefix}__{k}", v)

    # If model has a pipeline, log all step parameters recursively
    if hasattr(model, "pipeline") and isinstance(model.pipeline, Pipeline):
        for step_name, step in model.pipeline.named_steps.items():
            log_sklearn_estimator_params(step, prefix=f"{prefix}__{step_name}")


def xywh_to_xyxy(box):
    """
    Convert bounding box from (x, y, width, height) to (x_min, y_min, x_max, y_max)

    Args:
        box (tuple or list): (x, y, w, h)

    Returns:
        tuple: (x_min, y_min, x_max, y_max)
    """
    x, y, w, h = box
    return (x, y, x + w, y + h)


def xyxy_to_xywh(box):
    """
    Convert bounding box from (x_min, y_min, x_max, y_max) to (x, y, width, height)

    Args:
        box (tuple or list): (x_min, y_min, x_max, y_max)

    Returns:
        tuple: (x, y, w, h)
    """
    x_min, y_min, x_max, y_max = box
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def compute_detection_metrics(preds, y_true, iou_threshold=0.5):
    """
    Computes basic object detection metrics.

    Args:
        preds: List[List[Detection]] — predictions per image
        y_true: List[List[dict]] — COCO-style ground-truth per image
        iou_threshold: float — threshold to consider a prediction a match

    Returns:
        dict with precision, recall, f1_score, mean_iou
    """

    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / (boxAArea + boxBArea - interArea)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_scores = []

    for pred_boxes, true_anns in zip(preds, y_true):
        matched_gt = set()

        for pred in pred_boxes:
            pb = pred.bounding_box
            pred_label = pred.label

            if pred_label != "kite":
                continue  # only count kite predictions

            matched = False
            for i, ann in enumerate(true_anns):
                if i in matched_gt:
                    continue

                if ann["category_id"] != 1:
                    continue  # only count kite GTs (category_id: 1 in your dataset)

                tb = coco_to_xyxy(ann["bbox"])
                iou_score = iou(pb, tb)

                if iou_score >= iou_threshold:
                    matched = True
                    matched_gt.add(i)
                    total_tp += 1
                    iou_scores.append(iou_score)
                    break

            if not matched:
                total_fp += 1

        # Count false negatives (kite GTs not matched)
        total_fn += sum(1 for ann in true_anns if ann["category_id"] == 1) - len(
            matched_gt
        )

    # Final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
    }
