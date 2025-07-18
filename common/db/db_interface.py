from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
import sqlite3

from loguru import logger

from config.config_loader import load_config

CONFIG = load_config()

DB_PATH = Path(CONFIG["DB_PATH"])
QUERIES = Path(CONFIG["QUERIES"])
DETECTIONS_DIR = Path(CONFIG["DETECTIONS_DIR"])

import shutil


def setup_db():
    # Reset session folder on boot

    if DETECTIONS_DIR.exists():
        logger.warning("reset DETECTIONS_DIR")
        shutil.rmtree(DETECTIONS_DIR)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # make sure the db is setup correctly
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    create_image_metadata = (QUERIES / "create_image_metadata.sql").read_text()
    c.execute(create_image_metadata)
    create_serving_prediction = (QUERIES / "create_serving_prediction.sql").read_text()
    c.execute(create_serving_prediction)
    conn.close()
