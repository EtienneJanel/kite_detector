from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import FileResponse

load_dotenv()
from config.config_loader import load_config

CONFIG = load_config()
STATIC_DIR = Path(CONFIG["STATIC_DIR"])

router = APIRouter()


@router.get("")
@router.get("/")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.ico")
