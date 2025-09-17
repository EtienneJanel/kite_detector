# IDEA
mirror work done in https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/YOLOS/Fine_tuning_YOLOS_for_object_detection_on_custom_dataset_(balloon).ipynb#scrollTo=9r-lMAWKWoLY


# prepare dataser
```bash
python training/create_dataset_spot_detector.py \
    -o kite_detector/images/good \
    -z kite_detector/images/not_good \
    -p kite_detector/training/dataset/spot_fdt_labels_v0.0.3.csv
```

# run MlFlow server
`mlflow ui --port 5000`

# run trainer
the experiment name "spot fdt detector 0.0.1" will become model name "spot_fdt_detector_0.0.1.pkl"

```bash
python -m training.trainer \
    -m spot \
    -d training/dataset/spot_fdt_labels_v0.0.3.csv \
    -e "experiment name" \
    -r "run name" \
    -v "0.0.1"
```
or
```bash
python -m training.trainer \
    -m kite \
    -d training/dataset/fontedatelha-2.json \
    -e "kite_obj_detector" \
    -r "re-label to 38" \
    -v "0.0.3"
```

open tensorboard
`tensorboard --logdir training/models/kite_obj_detector_0.0.2/logs`


# Bbox format
https://www.learnml.io/posts/a-guide-to-bounding-box-formats/#different-box-formats

- COCO: XYWH
- yolo: XYWH

---

# Benchmark
Example Usage

```bash
python -m training.benchmark \
  -d training/dataset/fontedatelha-2.json \
  -e kite_obj_detection_benchmark \
  -r pretrained_yolos_tiny \
  --model_ckpt "hustvl/yolos-tiny"
```
```bash
python benchmark.py \
  -d datasets/kites/test_dataset.json \
  -e "Kite Detection" \
  -r "benchmark_finetuned_v1" \
  --fine_tuned_model_path training/models/Kite_Detection_v1
```


