#!/usr/bin/env python3
"""
Module to randomly adjust the contrast of an image using TensorFlow.
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Parameters:
    - image: 3D tf.Tensor representing the input image
    - lower: float, lower bound of the random contrast factor range
    - upper: float, upper bound of the random contrast factor range

    Returns:
    - The contrast-adjusted image tensor
    """
    return tf.image.random_contrast(image, lower, upper)
