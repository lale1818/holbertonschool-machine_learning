#!/usr/bin/env python3
"""
This module contains a function to calculate sensitivity for each class.
"""
import numpy as np


def sensitivity(confusion):
    """ Calculates the sensitivity for each class in a confusion matrix """
    tp = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    return tp / actual_positives
