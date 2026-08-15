#!/usr/bin/env python3
"""
Module to randomly crop an image using TensorFlow.
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Parameters:
    - image: 3D tf.Tensor containing the image to crop
    - size: tuple containing the shape of the crop (height, width, channels)

    Returns:
    - The randomly cropped image tensor
    """
    return tf.image.random_crop(image, size)
