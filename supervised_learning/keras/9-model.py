#!/usr/bin/env python3
"""
Save and Load Keras Model
"""
import tensorflow as tf


def save_model(network, filename):
    """
    Saves an entire model to a file.
    
    Args:
        network: The model to save.
        filename (str): The path of the file that the model should be saved to.
        
    Returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model from a file.
    
    Args:
        filename (str): The path of the file that the model should be loaded from.
        
    Returns:
        The loaded model.
    """
    return tf.keras.models.load_model(filename)
