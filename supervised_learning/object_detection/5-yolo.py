#!/usr/bin/env python3
"""
Module defining the Yolo class for object detection using YOLO v3.
"""
import cv2
import glob
import numpy as np
from tensorflow import keras as K


class Yolo:
    """
    Class Yolo that uses the Yolo v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.
        """
        self.model = K.models.load_model(model_path, compile=False)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes predictions from Darknet model for a single image.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h, image_w = image_size[0], image_size[1]
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_conf = 1 / (1 + np.exp(-output[..., 4:5]))
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))

            box_confidences.append(box_conf)
            box_class_probs.append(class_probs)

            c_x = np.tile(
                np.arange(grid_w).reshape(1, grid_w), (grid_h, 1)
            )
            c_x = np.tile(c_x[..., np.newaxis], (1, 1, num_anchors))

            c_y = np.tile(
                np.arange(grid_h).reshape(grid_h, 1), (1, grid_w)
            )
            c_y = np.tile(c_y[..., np.newaxis], (1, 1, num_anchors))

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            b_w = (anchor_w * np.exp(t_w)) / input_w
            b_h = (anchor_h * np.exp(t_h)) / input_h

            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes based on box score threshold.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            max_classes = np.argmax(scores, axis=-1)
            max_scores = np.max(scores, axis=-1)

            filtering_mask = max_scores >= self.class_t

            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(max_classes[filtering_mask])
            box_scores.append(max_scores[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression (NMS) to filtered bounding boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for c in unique_classes:
            idxs = np.where(box_classes == c)[0]

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h

                iou = inter / (areas[i] + areas[order[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            box_predictions.append(cls_boxes[keep])
            predicted_box_classes.append(np.full(len(keep), c))
            predicted_box_scores.append(cls_scores[keep])

        if len(box_predictions) > 0:
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0
            )
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0
            )
        else:
            box_predictions = np.empty((0, 4))
            predicted_box_classes = np.empty((0,))
            predicted_box_scores = np.empty((0,))

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a given folder path.
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = [cv2.imread(img_path) for img_path in image_paths]

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Resizes and rescales images for Darknet model input.

        Parameters:
        - images: list of images as numpy.ndarrays

        Returns:
        - tuple of (pimages, image_shapes):
          - pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
          - image_shapes: numpy.ndarray of shape (ni, 2) [h, w]
        """
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append([img.shape[0], img.shape[1]])

            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC
            )
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)
