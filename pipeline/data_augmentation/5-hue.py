#!/usr/bin/env python3
"""
Module to adjust the hue of an image using TensorFlow.
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Parameters:
    - image: 3D tf.Tensor containing the image to change
    - delta: float, the amount the hue should change [-1.0, 1.0]

    Returns:
    - The hue-adjusted image tensor
    """
    return tf.image.adjust_hue(image, delta)
