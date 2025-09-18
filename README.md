
# TO DO
- Docker
    - last build was 30min and 9GB
        - some libs are not used by serving (e.g. mlflow) and should be removed
        - side task: move training in another repo
- Serving
    - [BUG]:
        - dashboard: total kite count is incorrect (to investigate).
            ex: Total Captures: 2, but only 1 image. FDT works but not LDA.
        - dashboard: "Kite Ratio: 15.0%" doesn't make sense - to clarify/fix or remove.
    - [FEATURE]:
        - storage: read/write local DB and images to avoid storing unnecessary elements
        - dashboard: 
            - add swipe on images (phone)

- Training
    - port training in google-collab based on https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb#scrollTo=9r-lMAWKWoLY

# CHANGELOG
- Training
    - fine-tune on only 1 class
    - add Non-Maximum Suppression (NMS) to remove overlapping bounding boxes
    - add approximate of mAP (mean Average Precision) metric
    - add benchmark.py script
- Serving
    - fix date format (utc to gmt)

# Kite Detector
**Objective**: Open online beach-cam to detect flying kites in images, using a fine-tuned YOLO model. Provides summary of detections and images with bounding boxes on web-page. 
Have a rest end-point for home-assistant to send a notification to the user.

## Setup project
install dependencies
```bash
poetry shell
poetry install
```

## Serve app locally
```bash
make run
```

## End-points
- http://localhost:8000/home: main dashboard
- http://localhost:8000/health: health end point
- http://localhost:8000/binary_kite_sensor: home assistant end point
- http://localhost:8000/predict: prediction on a local image

**Predict**
```bash
GET http://localhost:8000/predict?image_path='serving/tests/assets/cbc9b24f-4ed9-4a69-98bd-c27bdb8bf0d7.jpg'
```

## Docker
### Build
```bash
make build
```
### Run
run docker with terminal
```bash
make compose
```

---
# Notes

## Yolo
- https://huggingface.co/docs/transformers/model_doc/yolos#transformers.YolosForObjectDetection
- https://huggingface.co/hustvl/yolos-tiny

## RT-DETR
- https://huggingface.co/docs/transformers/model_doc/rt_detr_v2

