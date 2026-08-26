#!/usr/bin/env python3
"""
Module to initialize the Q-table for a Gymnasium environment.
"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table with zeros.

    Parameters:
    - env: the FrozenLakeEnv instance

    Returns:
    - Q-table as a numpy.ndarray of zeros with shape (states, actions)
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    return q_table
