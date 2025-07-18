# serving/routes/results.py

import sqlite3
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from loguru import logger

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()

DB_PATH = CONFIG["DB_PATH"]

from common.utils import summarize_predictions

router = APIRouter()


@router.get("/")
def show_results(request: Request):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch predictions of the day
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
            date(s.created_date) >= current_date
            AND i.is_stored = TRUE
        ORDER BY s.created_date DESC
    """,
    )
    rows = cursor.fetchall()
    conn.close()

    max_stat_by_folder = summarize_predictions(
        rows, confidence_threshold=CONFIG["confidence_threshold"]
    )
    if not max_stat_by_folder:
        return {"binary_kite_sensor": False}

    for folder, label_summary_list in max_stat_by_folder.root.items():
        for label_summary in label_summary_list:
            if label_summary.label == "kite":
                return {"binary_kite_sensor": True}
    return {"binary_kite_sensor": False}
