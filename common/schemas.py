from typing import Dict, List

from pydantic import BaseModel, RootModel


class Detection(BaseModel):
    label: str
    confidence: float
    bounding_box: List[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    detections: List[Detection]


class LabelSummary(BaseModel):
    label: str
    max: int


class SummarizedPredictionsResponse(RootModel[Dict[str, List[LabelSummary]]]):
    def get(self, cam: str) -> List[LabelSummary]:
        return self.root.get(cam, {})


class CaptureMetadata(BaseModel):
    id: str
    timestamp: str
    bbox_url: str


class CapturesByFolder(RootModel[Dict[str, List[CaptureMetadata]]]):
    def get(self, cam: str) -> List[CaptureMetadata]:
        return self.__root__.get(cam, [])


class MetricsSummary(BaseModel):
    total: int
    kite_count: int
    kite_ratio: float
    capture_rate: float
