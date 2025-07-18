import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Query, Request
from fastapi.templating import Jinja2Templates
from loguru import logger

from common.io.annotate import draw_boxes_on_image
from common.schemas import (
    CaptureMetadata,
    CapturesByFolder,
    DetectionResponse,
    MetricsSummary,
)

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()

DETECTIONS_DIR = Path(CONFIG["DETECTIONS_DIR"])
IMAGES_DIR = Path(CONFIG["IMAGES_DIR"])
DB_PATH = Path(CONFIG["DB_PATH"])
confidence_threshold = CONFIG["confidence_threshold"]

from common.utils import summarize_predictions

router = APIRouter()
templates = Jinja2Templates(directory="serving/templates")


@router.get("/")
def show_results(request: Request, minutes: int = Query(30, gt=0, le=1440)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch predictions from the last 30 minutes
    cutoff_time = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
    cursor.execute(
        """
        SELECT 
            s.id,
            s.created_date,
            s.predictions,
            i.folder
        FROM serving_prediction s
        JOIN image_metadata i
            ON i.id = s.id
        WHERE 
            s.created_date >= ?
            AND i.is_stored = TRUE
        ORDER BY s.created_date DESC
    """,
        (cutoff_time,),
    )
    rows = cursor.fetchall()
    conn.close()
    logger.info(rows)

    captures_dict = defaultdict(list)
    kite_count = 0

    max_stat_by_folder = summarize_predictions(rows)

    captures_by_folder = {}
    for row in rows:
        image_id, timestamp, preds_json, folder = row
        if not captures_by_folder.get(folder):
            captures_by_folder[folder] = []
        try:
            preds = DetectionResponse.model_validate_json(preds_json)
        except:
            preds = DetectionResponse(detections=[])

        # Count kites
        kite_count += sum(
            1
            for obj in preds.detections
            if (obj.label == "kite" and obj.confidence >= confidence_threshold)
        )

        original_img = IMAGES_DIR / folder / f"{image_id}.jpg"
        annotated_img = DETECTIONS_DIR / folder / f"{image_id}.jpg"

        if not annotated_img.exists():
            try:
                logger.info("Drawing squares...")
                draw_boxes_on_image(
                    original_img,
                    annotated_img,
                    preds.detections,
                    confidence_threshold=confidence_threshold,
                )
            except Exception as e:
                logger.error(f"❌ Error drawing boxes: {e}")

        captures_dict[folder].append(
            CaptureMetadata(
                id=image_id,
                timestamp=timestamp,
                bbox_url=f"/detections/{folder}/{image_id}.jpg",
            )
        )

    captures_by_folder = CapturesByFolder(captures_dict)

    metrics = MetricsSummary(
        total=len(rows),
        kite_count=kite_count,
        kite_ratio=(kite_count / len(rows) * 100) if rows else 0.0,
        capture_rate=round(len(rows) / 30.0, 2),
    )

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "captures_by_folder": captures_by_folder.model_dump(),
            "max_stat_by_folder": max_stat_by_folder.model_dump(),
            "metrics": metrics.model_dump(),
        },
    )
