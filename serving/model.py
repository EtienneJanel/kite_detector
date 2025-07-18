from pathlib import Path
from typing import Union

import torch
from PIL import Image
from transformers import YolosForObjectDetection, YolosImageProcessor

from common.schemas import Detection, DetectionResponse

# Load model and processor once
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


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
        threshold=0.5,
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
