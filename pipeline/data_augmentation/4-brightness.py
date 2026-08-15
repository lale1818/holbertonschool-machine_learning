#!/usr/bin/env python3
"""
Module to randomly adjust the brightness of an image using TensorFlow.
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Parameters:
    - image: 3D tf.Tensor containing the image to change
    - max_delta: float, maximum amount the image brightness can change

    Returns:
    - The brightness-adjusted image tensor
    """
    return tf.image.random_brightness(image, max_delta)
