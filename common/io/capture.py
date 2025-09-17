import hashlib
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv
from loguru import logger

from training.model import SpotDetector

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()

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
model_path = CONFIG["model_path"]


def capture_single_image(video_url: str, folder: str, model_predict_fn):
    """
    Captures a single frame from the webcam stream, saves it to disk,
    runs predictions on the image, and stores the results in the database.

    Args:
        video_url (str): The URL of the video stream.
        folder (str): The folder where the image will be saved.
        model_predict_fn (Callable): The prediction function to apply on the image.
    """
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"Failed to open stream: {video_url}")
        return

    success, frame = cap.read()
    cap.release()

    if not success:
        logger.warning("Could not capture frame.")
        return

    # Hash-based ID
    image_id = hashlib.sha256(frame.tobytes()).hexdigest()
    timestamp = datetime.utcnow().isoformat()
    # frame_resized = cv2.resize(frame, (780, 540), interpolation=cv2.INTER_LINEAR)

    coco_path = IMAGES_DIR / folder
    coco_path.mkdir(parents=True, exist_ok=True)
    img_path = coco_path / f"{image_id}.jpg"
    cv2.imwrite(str(coco_path / f"{image_id}.jpg"), frame)

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


def capture_oriented_frame(
    video_url: str,
    folder,
    model_predict_fn,
    buffer_duration_sec: str = 60,
    frame_interval_sec: str = 2,
):
    """
    Captures frames from the webcam over a buffer period, then selects the
    first frame with the correct orientation based on model prediction.
    Saves the selected frame, runs object prediction, and logs results.

    Args:
        video_url (str): The URL of the video stream.
        folder (str): The folder where the image will be saved.
        model_predict_fn (Callable): The prediction function to apply on the image.
        buffer_duration_sec (int): Duration (in seconds) to collect frames.
        frame_interval_sec (int): Interval (in seconds) between frame captures.
    """
    model_spot_detector = SpotDetector()
    model_spot_detector.load_model(model_path)

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"Failed to open stream: {video_url}")
        return

    frames = []  # raw data of image
    timestamps = []

    start_time = time.time()
    logger.info(f"start buffering {buffer_duration_sec}")
    while time.time() - start_time < buffer_duration_sec:
        success, frame = cap.read()
        if success:
            frames.append(frame)
            timestamps.append(datetime.utcnow().isoformat())
        time.sleep(frame_interval_sec)

    cap.release()

    if not frames:
        logger.warning("No frames captured in buffer.")
        return

    # Choose the first "good" frame
    for frame, timestamp in zip(frames, timestamps):
        label = model_spot_detector.predict([frame])[0]
        if label == 1:
            image_id = hashlib.sha256(frame.tobytes()).hexdigest()
            final_path = IMAGES_DIR / folder / f"{image_id}.jpg"
            Path(final_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(final_path)
            cv2.imwrite(str(final_path), frame)

            # Predict object (YOLO) and save
            predictions = model_predict_fn(str(final_path))

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            insert_pred = (QUERIES / "insert_serving_prediction.sql").read_text()
            cursor.execute(insert_pred, (image_id, timestamp, predictions.json()))
            conn.commit()

            insert_metadata = (QUERIES / "insert_image_metadata.sql").read_text()
            cursor.execute(
                insert_metadata,
                (image_id, timestamp, folder, KEEP_IMAGE_FOR_TRAINING),
            )
            conn.commit()
            conn.close()

            logger.info(f"✅ Captured good frame from {folder}")
            break
    else:
        logger.warning(f"🚫 No valid orientation detected in buffer from {folder}")
