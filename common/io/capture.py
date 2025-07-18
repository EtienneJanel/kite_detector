import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()

DETECTIONS_DIR = Path(CONFIG["DETECTIONS_DIR"])
IMAGES_DIR = Path(CONFIG["IMAGES_DIR"])
DB_PATH = Path(CONFIG["DB_PATH"])
QUERIES = Path(CONFIG["QUERIES"])

KEEP_IMAGE_FOR_TRAINING = CONFIG["KEEP_IMAGE_FOR_TRAINING"]
DICT_VIDEO_URL = CONFIG["DICT_VIDEO_URL"]
end_capture_hour = CONFIG["end_capture_hour"]
interval_capture_between_loops_sec = CONFIG["interval_capture_between_loops_sec"]
number_loops = CONFIG["number_loops"]
sleep_sec = CONFIG["sleep_sec"]
start_capture_hour = CONFIG["start_capture_hour"]


def capture_single_image(video_url, folder, model_predict_fn):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"Failed to open stream: {video_url}")
        return

    success, frame = cap.read()
    cap.release()

    # if os.getenv("APP_ENV") == "prod":
    # TODO: add a buffer of images
    #     # update this part
    #     reference_folder = (
    #         IMAGES_DIR / "fontedatelha_0_references"
    #     )  # 5 images reference that we don't want
    #     pass

    if not success:
        logger.warning("Could not capture frame.")
        return

    # Hash-based ID
    image_id = hashlib.sha256(frame.tobytes()).hexdigest()
    timestamp = datetime.utcnow().isoformat()
    frame_resized = cv2.resize(frame, (780, 540), interpolation=cv2.INTER_LINEAR)

    coco_path = IMAGES_DIR / folder
    coco_path.mkdir(parents=True, exist_ok=True)
    img_path = coco_path / f"{image_id}.jpg"
    cv2.imwrite(str(coco_path / f"{image_id}.jpg"), frame_resized)

    # Predict
    predictions = model_predict_fn(str(img_path))

    # Save prediction in persistent table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    insert_serving_prediction = (QUERIES / "insert_serving_prediction.sql").read_text()
    cursor.execute(insert_serving_prediction, (image_id, timestamp, predictions.json()))
    conn.commit()

    insert_image_metadata = (QUERIES / "insert_image_metadata.sql").read_text()
    cursor.execute(
        insert_image_metadata,
        (image_id, timestamp, folder, KEEP_IMAGE_FOR_TRAINING),
    )
    conn.commit()

    conn.close()

    logger.info(f"Captured + processed image from {folder}")
