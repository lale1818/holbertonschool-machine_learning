#!/usr/bin/env python3
"""
This module contains a function to calculate precision for each class.
"""
import numpy as np


def precision(confusion):
    """ Calculates the precision for each class in a confusion matrix """
    tp = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return tp / predicted_positives
