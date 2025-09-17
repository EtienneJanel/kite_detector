import asyncio
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from config.config_loader import load_config

CONFIG = load_config()

DICT_VIDEO_URL = CONFIG["DICT_VIDEO_URL"]
end_capture_hour = CONFIG["end_capture_hour"]
interval_sleep_between_cam_sec = CONFIG["interval_sleep_between_cam_sec"]
sleep_sec = CONFIG["sleep_sec"]
buffer_duration_sec = CONFIG["buffer_duration_sec"]
frame_interval_sec = CONFIG["frame_interval_sec"]
start_capture_hour = CONFIG["start_capture_hour"]

from common.io.capture import capture_oriented_frame, capture_single_image
from serving.model import predict  # the model inference function

# 🔒 Global lock for concurrency control
capture_lock = asyncio.Lock()


async def safe_capture(
    video_url: str,
    folder: str,
    model_predict_fn,
    buffer_duration_sec: int,
    frame_interval_sec: int,
):
    """
    Executes the frame capture function safely using a concurrency lock.

    Ensures that only one instance of the capture task runs at any given time
    by using an asyncio.Lock. If a capture is already in progress, the function
    will skip execution and return immediately.

    Args:
        video_url (str): The video stream URL to capture from.
        folder (str): The folder name for saving captured images.
        model_predict_fn (Callable): The prediction function to apply on captured images.
        buffer_duration_sec (int): Duration (in seconds) to buffer frames during capture.

    Returns:
        None
    """
    if capture_lock.locked():
        return
    async with capture_lock:
        await asyncio.to_thread(
            capture_oriented_frame,
            video_url=video_url,
            folder=folder,
            model_predict_fn=model_predict_fn,
            buffer_duration_sec=buffer_duration_sec,
            frame_interval_sec=frame_interval_sec,
        )


async def run_capture_loop():
    """
    Main background loop for scheduled webcam capture.

    Periodically checks the current time and triggers frame capture
    for each configured video stream if within the allowed capture hours.
    Captures are spaced using configurable sleep intervals, and concurrency
    is managed to ensure only one capture task runs at a time.

    This function is designed to run as a long-lived async task, such as
    in a FastAPI startup event or background service.

    Returns:
        None
    """
    nb_spots = len(DICT_VIDEO_URL)
    total_hours_in_sec = (end_capture_hour - datetime.now().hour) * 3600

    logger.info(
        f"""📷 Started background capture task.
capture params:
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
                if folder == "fontedatelha":
                    asyncio.create_task(
                        safe_capture(
                            video_url=url,
                            folder=folder,
                            model_predict_fn=predict,
                            buffer_duration_sec=buffer_duration_sec,
                            frame_interval_sec=frame_interval_sec,
                        )
                    )
                else:
                    capture_single_image(
                        video_url=url,
                        folder=folder,
                        model_predict_fn=predict,
                    )
                await asyncio.sleep(interval_sleep_between_cam_sec)
        await asyncio.sleep(sleep_sec)
