#!/usr/bin/env python3
"""
This module contains a function to create a confusion matrix.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Creates a confusion matrix from one-hot encoded labels and logits """
    return np.matmul(labels.T, logits)
