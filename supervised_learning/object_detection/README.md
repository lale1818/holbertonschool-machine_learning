# Object Detection

This directory contains implementations for object detection using the YOLO v3 algorithm in TensorFlow and Keras.

## Requirements
- Python 3.9
- TensorFlow 2.x
- NumPy
- `pycodestyle` 2.11.1

## Files
- `0-yolo.py`: Defines the `Yolo` class initializing model path, class names, thresholds, and anchor boxes.
- `1-yolo.py`: Adds `process_outputs` to decode bounding box coordinates and confidences.
- `2-yolo.py`: Adds `filter_boxes` to remove low-confidence predictions.
- `3-yolo.py`: Adds `non_max_suppression` to remove overlapping bounding boxes.
