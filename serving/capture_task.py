import asyncio
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()

DETECTIONS_DIR = CONFIG["DETECTIONS_DIR"]
IMAGES_DIR = CONFIG["IMAGES_DIR"]
DB_PATH = CONFIG["DB_PATH"]
QUERIES = CONFIG["QUERIES"]
DICT_VIDEO_URL = CONFIG["DICT_VIDEO_URL"]
end_capture_hour = CONFIG["end_capture_hour"]
interval_sleep_between_cam_sec = CONFIG["interval_sleep_between_cam_sec"]
sleep_sec = CONFIG["sleep_sec"]
start_capture_hour = CONFIG["start_capture_hour"]

from common.io.capture import capture_single_image
from serving.model import predict  # the model inference function


async def run_capture_loop():
    logger.info("✅ Started background capture task.")
    nb_spots = len(DICT_VIDEO_URL)
    total_hours_in_sec = (end_capture_hour - datetime.now().hour) * 60 * 60
    logger.info(
        f"""capture params:
- start_capture_hour: {start_capture_hour}
- end_capture_hour: {end_capture_hour}
- interval_sleep_between_cam_sec: {interval_sleep_between_cam_sec}
- sleep_sec: {sleep_sec}
- total expected images between now and end: {int(total_hours_in_sec / sleep_sec * nb_spots)}
    """
    )

    while True:
        now = datetime.now()
        if start_capture_hour <= now.hour <= end_capture_hour:
            for folder, url in DICT_VIDEO_URL.items():
                capture_single_image(
                    video_url=url,
                    folder=folder,
                    model_predict_fn=predict,
                )
                await asyncio.sleep(interval_sleep_between_cam_sec)
        await asyncio.sleep(sleep_sec)
