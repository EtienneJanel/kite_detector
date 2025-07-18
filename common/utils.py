from collections import defaultdict

import pandas as pd
from loguru import logger

from common.schemas import (
    DetectionResponse,
    LabelSummary,
    SummarizedPredictionsResponse,
)


def summarize_predictions(
    rows: list[dict], confidence_threshold=0.8
) -> SummarizedPredictionsResponse:
    if not rows:
        return SummarizedPredictionsResponse({})

    df = pd.DataFrame(rows, columns=["id", "timestamp", "predictions", "folder"])

    summary_data = []

    for _, row in df.iterrows():
        try:
            prediction_obj = DetectionResponse.model_validate_json(row["predictions"])
        except Exception as e:
            logger.warning(f"Invalid prediction JSON for row {row['id']}: {e}")
            continue

        label_counts = {}
        for detection in prediction_obj.detections:
            if detection.confidence >= confidence_threshold:
                label = detection.label
                label_counts[label] = label_counts.get(label, 0) + 1

        for label, count in label_counts.items():
            summary_data.append(
                {
                    "id": row["id"],
                    "folder": row["folder"],
                    "label": label,
                    "count": count,
                }
            )

    if not summary_data:
        return SummarizedPredictionsResponse({})

    summary_df = pd.DataFrame(summary_data)

    # Max count per (folder, label)
    summary_df_max_by_folder_label = (
        summary_df.groupby(["folder", "label"])["count"]
        .max()
        .reset_index()
        .rename(columns={"count": "max"})
    )

    max_stat_by_folder = defaultdict(list)
    for _, row in summary_df_max_by_folder_label.iterrows():
        folder = row["folder"]
        entry = LabelSummary(label=row["label"], max=row["max"])
        max_stat_by_folder[folder].append(entry)

    return SummarizedPredictionsResponse(max_stat_by_folder)
