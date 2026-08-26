#!/usr/bin/env python3
"""
Module defining the Yolo class for object detection using YOLO v3.
"""
from tensorflow import keras as K


class Yolo:
    """
    Class Yolo that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.

        Parameters:
        - model_path: path to where a Darknet Keras model is stored
        - classes_path: path to list of class names used for the Darknet model
        - class_t: float representing the box score threshold
        - nms_t: float representing the IOU threshold for non-max suppression
        - anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        self.model = K.models.load_model(model_path, compile=False)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
