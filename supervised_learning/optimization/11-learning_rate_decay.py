#!/usr/bin/env python3
"""
This module contains a function that updates the learning rate
using inverse time decay in NumPy.
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Parameters:
    - alpha: original learning rate
    - decay_rate: weight used to determine the rate at which alpha decays
    - global_step: number of passes of gradient descent that have elapsed
    - decay_step: number of passes before alpha is decayed further

    Returns:
    - updated value for alpha
    """
    step = global_step // decay_step
    alpha_updated = alpha / (1 + decay_rate * step)

    return alpha_updated
