import asyncio
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()
from common.db.db_interface import setup_db
from config.config_loader import load_config

CONFIG = load_config()

DETECTIONS_DIR = Path(CONFIG["DETECTIONS_DIR"])
IMAGES_DIR = Path(CONFIG["IMAGES_DIR"])
DB_PATH = Path(CONFIG["DB_PATH"])
QUERIES = Path(CONFIG["QUERIES"])

from serving.capture_task import run_capture_loop
from serving.routes.binary_kite_sensor import router as binary_kite_sensor_router
from serving.routes.health import router as health_router
from serving.routes.home import router as home_router
from serving.routes.predict import router as predict_router

app = FastAPI()

app.include_router(home_router, prefix="/home", tags=["home"])
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(predict_router, prefix="/predict", tags=["predict"])
app.include_router(
    binary_kite_sensor_router,
    prefix="/binary_kite_sensor",
    tags=["binary_kite_sensor"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/detections", StaticFiles(directory=DETECTIONS_DIR), name="detections")


@app.on_event("startup")
async def startup_event():
    setup_db()
    # start capture
    asyncio.create_task(run_capture_loop())


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
