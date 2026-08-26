#!/usr/bin/env python3
"""
Save and Load Keras Model
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire model to a file.

    Args:
        network: The model to save.
        filename: Path of the file that the model should be saved to.

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model from a file.

    Args:
        filename: Path of the file that the model should be loaded from.

    Returns:
        The loaded model.
    """
    return K.models.load_model(filename)
