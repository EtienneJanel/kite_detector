from collections import defaultdict

import numpy as np
import torch
from loguru import logger
from torchvision.ops import box_iou
from tqdm import tqdm

# --- Helper conversions ---


def coco_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def detections_to_xyxy(preds, confidence=0.5, label="kite"):
    """
    Convert list of Detection objects to xyxy boxes + scores.
    """
    boxes = []
    scores = []
    for obj in preds:
        if obj.label == label and obj.confidence >= confidence:
            boxes.append(obj.bounding_box)
            scores.append(obj.confidence)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
        scores, dtype=torch.float32
    )


def groundtruth_to_xyxy(trues, label_id=1):
    """
    Convert COCO-style annotations to xyxy boxes.
    """
    boxes = []
    for ann in trues:
        # logger.info(ann)
        if ann.get("category_id", 0) == label_id:
            boxes.append(coco_xywh_to_xyxy(ann["bbox"]))
    return torch.tensor(boxes, dtype=torch.float32)


# --- Core metric calculation ---


def evaluate_model(model, X, y_true, confidence=0.5, iou_threshold=0.5):
    """
    Evaluate a model on dataset X, y_true.
    Returns dict with precision, recall, f1, and mAP@IoU.
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    aps = []

    for i in tqdm(range(len(X)), desc="Evaluating"):
        # Run inference
        preds = model.predict([X[i]])[0]  # list of Detection objects

        # Convert to tensors
        pred_boxes, pred_scores = detections_to_xyxy(preds, confidence=confidence)
        gt_boxes = groundtruth_to_xyxy(y_true[i])
        # logger.info(pred_boxes)
        # logger.info(gt_boxes)

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        elif len(gt_boxes) == 0 and len(pred_boxes) > 0:
            all_fp += len(pred_boxes)
            continue
        elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
            all_fn += len(gt_boxes)
            continue

        # IoU matrix
        ious = box_iou(pred_boxes, gt_boxes)

        # Match predictions to GT
        matched_gt = set()
        tp, fp = 0, 0
        for pred_idx in range(len(pred_boxes)):
            max_iou, gt_idx = torch.max(ious[pred_idx], dim=0)
            if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
                tp += 1
                matched_gt.add(gt_idx.item())
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)

        all_tp += tp
        all_fp += fp
        all_fn += fn

        # AP calculation (simplified: AP = TP / (TP+FP+FN) per image)
        denom = tp + fp + fn
        if denom > 0:
            aps.append(tp / denom)

    # Global metrics
    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mAP = np.mean(aps) if aps else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "mAP": mAP}


def benchmark_models(models: dict, X, y_true, confidence=0.5, iou_threshold=0.5):
    """
    Benchmark multiple models on the same dataset.

    Args:
        models (dict): {"name": model_object, ...}
        X (list): list of image paths
        y_true (list): list of ground truth annotations (COCO-style)
        confidence (float): minimum confidence threshold for predictions
        iou_threshold (float): IoU threshold for TP/FP assignment

    Returns:
        dict: {"model_name": {"precision":..., "recall":..., "f1":..., "mAP":...}, ...}
    """
    results = {}
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        results[name] = evaluate_model(model, X, y_true, confidence, iou_threshold)

    # Print table
    metrics = list(next(iter(results.values())).keys())
    print("\n=== Benchmark Results ===")
    print(f"{'Metric':<12} | " + " | ".join([f"{name:<12}" for name in results.keys()]))
    print("-" * (15 + len(results) * 15))
    for metric in metrics:
        row = f"{metric:<12} | " + " | ".join(
            [f"{results[name][metric]:<12.4f}" for name in results.keys()]
        )
        print(row)

    return results
