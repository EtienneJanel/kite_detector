import json
import logging
from pathlib import Path

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
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)

from common.schemas import Detection, DetectionResponse


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


# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------


class KiteImagePreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer for converting images into tensor input.
    Doesn't handle annotations.
    """

    def __init__(self, model_ckpt="hustvl/yolos-tiny"):
        self.model_ckpt = model_ckpt
        self.processor = None

    def fit(self, X, y=None):
        from transformers import AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(self.model_ckpt)
        return self

    def transform(self, X):
        transformed = []

        for idx, item in enumerate(X):
            if isinstance(item, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
            elif isinstance(item, (str, Path)):
                image = Image.open(item).convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(item)}")

            processed = self.processor(images=image, return_tensors="pt")
            processed["original_size"] = (image.height, image.width)
            processed["image_path"] = str(item)
            transformed.append(processed)

        return transformed


class KiteModel(BaseEstimator):
    """
    Final estimator that trains and predicts bounding boxes using a HuggingFace model.
    """

    def __init__(
        self,
        model_ckpt="hustvl/yolos-tiny",
        threshold=0.5,
        training_log_path: Path = None,
        num_labels=1,
    ):
        self.model_ckpt = model_ckpt
        self.threshold = threshold
        self.training_log_path = training_log_path
        self.num_labels = num_labels

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def _load_model(self):
        from transformers import AutoImageProcessor

        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_ckpt,
            # num_labels=self.num_labels,
        ).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(self.model_ckpt)
        self.model.eval()

    def xywh_to_xyxy_scaled(
        self, x, y, w, h, image_tensor, original_height, original_width
    ):
        input_h, input_w = image_tensor.shape[1:]  # (H, W)

        # Scaling factors
        scale_x = input_w / original_width
        scale_y = input_h / original_height

        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        # Scale to match YOLOS input
        x_min *= scale_x
        x_max *= scale_x
        y_min *= scale_y
        y_max *= scale_y

        return x_min, y_min, x_max, y_max

    def fit(self, X, y):
        if self.model is None:
            self._load_model()

        # Format the dataset
        formatted_data = []
        for i, image_data in enumerate(X):
            # Get processed pixel_values
            image_tensor = image_data["pixel_values"].squeeze(0)
            original_height, original_width = image_data["original_size"]

            coco_anns = y[i]
            boxes = []
            class_labels = []
            areas = []
            image_ids = []
            iscrowds = []

            for ann in coco_anns:
                image_ids.append(ann["image_id"])
                x, y_, w, h = ann["bbox"]
                x_min, y_min, x_max, y_max = self.xywh_to_xyxy_scaled(
                    x, y_, w, h, image_tensor, original_height, original_width
                )

                boxes.append([x_min, y_min, x_max, y_max])

                class_labels.append(ann["category_id"])
                areas.append(ann["area"])
                iscrowds.append(ann["iscrowd"])
                # logger.info(f"Original size: {original_width}x{original_height}")
                # logger.info(f"Input size (after processor): {image_tensor.shape[1:]}")  # HxW
                # logger.info(f"First bbox (scaled): {boxes[0]}")
                # logger.info(f"Scaled bbox: {[x_min, y_min, x_max, y_max]}")
                # break



            num_objs = len(class_labels)

            target = {
                "image_id": torch.tensor(image_ids[0], dtype=torch.int64),
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "class_labels": torch.tensor(class_labels, dtype=torch.int64),
                "area": torch.tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
            }

            formatted_data.append(
                {
                    "pixel_values": image_tensor,
                    "labels": target,
                }
            )

        # Define a dataset
        class CustomYOLOSDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, idx):
                return self.data[idx]

            def __len__(self):
                return len(self.data)

        dataset = CustomYOLOSDataset(formatted_data)

        # Define training arguments
        args = TrainingArguments(
            output_dir=self.training_log_path,
            per_device_train_batch_size=4,
            num_train_epochs=10,
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,  # Adjust based on dataset size
            logging_dir=self.training_log_path / "logs",
            report_to=["tensorboard"],  # <--- Enable this
        )

        # Custom collate function (not DefaultDataCollator!)
        def collate_fn(batch):
            pixel_values = [item["pixel_values"] for item in batch]
            labels = [item["labels"] for item in batch]
            return {"pixel_values": torch.stack(pixel_values), "labels": labels}

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            data_collator=collate_fn,
        )

        trainer.train()
        self.save_model(self.training_log_path)
        return self

    def predict(self, X):
        """
        Expects preprocessed input from the pipeline (i.e., list of dicts).
        Returns list of detections.
        """
        if self.model is None or self.processor is None:
            self._load_model()

        from common.schemas import Detection

        results = []

        for entry in X:
            pixel_values = entry["pixel_values"].to(self.device)
            target_size = [entry["original_size"]]
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)

            processed = self.processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor(target_size, device=self.device),
                threshold=self.threshold,
            )

            detections = []
            for result in processed:
                for score, label_id, box in zip(
                    result["scores"], result["labels"], result["boxes"]
                ):
                    detections.append(
                        Detection(
                            label=self.model.config.id2label[label_id.item()],
                            confidence=round(score.item(), 2),
                            bounding_box=[round(c, 2) for c in box.tolist()],
                        )
                    )
            results.append(detections)

        return results

    def _debug(self, X):
        if self.model is None or self.processor is None:
            self._load_model()

        from common.schemas import Detection

        results = []

        for entry in X:
            pixel_values = entry["pixel_values"].to(self.device)
            target_size = [entry["original_size"]]
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)

            results.append(outputs)

        return results

    def save_model(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    def load_model(self, path):
        from transformers import AutoImageProcessor

        self.model = AutoModelForObjectDetection.from_pretrained(path).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model.eval()
        return self


class KiteDetector(BaseEstimator):
    """
    Sklearn-compatible high-level model with preprocessing + object detection pipeline.
    """

    def __init__(self, model_ckpt="hustvl/yolos-tiny", training_log_path: Path = None):
        self.model_ckpt = model_ckpt
        self.training_log_path = training_log_path
        self.pipeline = Pipeline(
            [
                ("preprocessor", KiteImagePreprocessor(model_ckpt=model_ckpt)),
                (
                    "detector",
                    KiteModel(
                        model_ckpt=model_ckpt, training_log_path=training_log_path
                    ),
                ),
            ]
        )

    def fit(self, X, y):
        return self.pipeline.fit(X, y)

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError(
                "Model is not trained or loaded. Call load_model() first."
            )
        return self.pipeline.predict(X)

    def save_model(self):
        """
        Save pipeline config and HuggingFace weights separately.
        """
        # Save config
        config = {
            "model_ckpt": self.model_ckpt,
            "training_log_path": str(self.training_log_path),
        }
        with open(
            self.training_log_path / f"{self.training_log_path.stem}.json", "w"
        ) as f:
            json.dump(config, f)

        # Save HuggingFace model
        detector = self.pipeline.named_steps["detector"]
        detector.model.save_pretrained(self.training_log_path)
        detector.processor.save_pretrained(self.training_log_path)

        # Save sklearn pipeline wrapper without the model
        self.pipeline.named_steps["detector"].model = None
        self.pipeline.named_steps["detector"].processor = None
        joblib.dump(self, self.training_log_path / f"{self.training_log_path.stem}.pkl")

    @classmethod
    def load_model(cls, directory_path=None):
        if directory_path is None:
            # Fallback to default YOLOS model
            model_ckpt = "hustvl/yolos-tiny"
            
            # Create detector and load pretrained model
            detector = KiteModel(model_ckpt=model_ckpt)
            detector._load_model()  # This sets detector.model and detector.processor

            # Create the full pipeline manually
            preprocessor = KiteImagePreprocessor(model_ckpt=model_ckpt)
            preprocessor.fit([])  # Force loading AutoImageProcessor

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("detector", detector),
            ])

            instance = cls(model_ckpt=model_ckpt)
            instance.pipeline = pipeline
            return instance
        
        directory_path = Path(directory_path)

        # Load the pipeline (sklearn part)
        pkl_path = directory_path / f"{directory_path.stem}.pkl"
        model = joblib.load(pkl_path)

        # Load custom config (contains training_log_path, etc.)
        config_path = directory_path / f"{directory_path.stem}.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Restore Hugging Face model and processor
        detector = model.pipeline.named_steps["detector"]
        detector.model = AutoModelForObjectDetection.from_pretrained(
            directory_path, local_files_only=True
        )
        detector.processor = AutoImageProcessor.from_pretrained(
            directory_path, local_files_only=True
        )

        return model
