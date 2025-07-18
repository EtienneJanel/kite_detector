from fastapi import APIRouter, Query

from common.schemas import DetectionResponse
from serving.model import predict

router = APIRouter()


@router.get("/", response_model=DetectionResponse)
def run_inference(image_path: str = Query(..., description="Path to the image file")):
    detections = predict(image_path)
    return detections.model_dump()
