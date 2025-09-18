from pathlib import Path
from typing import Union

import cv2
import joblib
import numpy as np
import torch
from loguru import logger
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import YolosForObjectDetection, YolosImageProcessor

from common.schemas import Detection, DetectionResponse

# Load model and processor once
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer to extract image features using a pretrained CNN (ResNet18 by default).
    """

    def __init__(self, model_name="resnet18", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _load_model(self):
        weights = ResNet18_Weights.DEFAULT
        base_model = resnet18(weights=weights)
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval().to(self.device)

        self.image_preprocessor = weights.transforms()

    def fit(self, X, y=None):
        self._load_model()
        return self

    def transform(self, X):
        """
        X: list of image file paths
        Returns: numpy array of extracted features
        """
        if self.model is None:
            self._load_model()
        features = []
        for item in X:
            try:
                if isinstance(item, np.ndarray):
                    image = Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
                elif isinstance(item, (str, Path)):
                    image = Image.open(item).convert("RGB")
                else:
                    raise ValueError(f"Unsupported input type: {type(item)}")

                tensor = self.image_preprocessor(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(tensor).squeeze().cpu().numpy()
                features.append(output)
            except Exception as e:
                logger.warning(f"[Warning] Failed to process item: {e}")
                features.append(np.zeros(512))

        return np.array(features)


class SpotDetector(BaseEstimator):
    """
    High-level model wrapper for training & inference.
    Sklearn-compatible and includes feature extraction + classification pipeline.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("feature_extractor", ImageFeatureExtractor()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )

    def load_model(self, model_path):
        self.pipeline = joblib.load(model_path)
        return self

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        X: list of image paths or list of OpenCV images (np.ndarray)
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)


def predict(image_path: Union[str, Path]) -> DetectionResponse:
    """
    Run object detection on a single image using the YOLOS-tiny model.

    Args:
        image_path (Union[str, Path]): Path to the image file.

    Returns:
        DetectionResponse: A structured list of detections, including label, confidence, and bounding box.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    raw_results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=0.6,
    )

    detections = []
    for result in raw_results:
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            detection = Detection(
                label=model.config.id2label[label_id.item()],
                confidence=round(score.item(), 2),
                bounding_box=[round(coord, 2) for coord in box.tolist()],
            )
            detections.append(detection)

    return DetectionResponse(detections=detections)
