# serving/routes/results.py

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()
DB_PATH = CONFIG["DB_PATH"]
BINARY_KITE_SENSOR_CACHE_PATH = Path(CONFIG["BINARY_KITE_SENSOR_CACHE_PATH"])
HTML_TEMPLATES_DIR = Path(CONFIG["HTML_TEMPLATES_DIR"])

from common.utils import summarize_predictions

router = APIRouter()

templates = Jinja2Templates(directory=HTML_TEMPLATES_DIR)


def is_cache_valid(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    with open(cache_path, "r") as f:
        cache = json.load(f)
        cache_date = datetime.strptime(cache["date"], "%Y-%m-%d").date()
        return cache_date == datetime.utcnow().date()


def read_cache(cache_path: Path) -> bool:
    with open(cache_path, "r") as f:
        return json.load(f)["binary_kite_sensor"]


def write_cache(cache_path: Path, value: bool):
    cache_data = {
        "binary_kite_sensor": value,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)


def binary_kite_sensor_html_template_response(request, payload: dict):
    if "text/html" in request.headers.get("accept", ""):
        return templates.TemplateResponse(
            "binary_kite_sensor.html",
            {
                "request": request,
                "sensor_data": payload,
            },
        )

    return JSONResponse(content=payload)


@router.get(
    "", response_class=HTMLResponse
)  # Handle both with and without trailing slash
def show_results(request: Request):
    if is_cache_valid(BINARY_KITE_SENSOR_CACHE_PATH):
        logger.debug("Returning value from cache.")
        return binary_kite_sensor_html_template_response(
            request, {"binary_kite_sensor": read_cache(BINARY_KITE_SENSOR_CACHE_PATH)}
        )

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
        return binary_kite_sensor_html_template_response(
            request, {"binary_kite_sensor": False}
        )

    for folder, label_summary_list in max_stat_by_folder.root.items():
        for label_summary in label_summary_list:
            if label_summary.label == "kite":
                write_cache(BINARY_KITE_SENSOR_CACHE_PATH, True)
                return binary_kite_sensor_html_template_response(
                    request, {"binary_kite_sensor": True}
                )
    return binary_kite_sensor_html_template_response(
        request, {"binary_kite_sensor": False}
    )
