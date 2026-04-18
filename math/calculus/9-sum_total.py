#!/usr/bin/env python3
"""
Module to calculate the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n
    """
    # type(n) is not int yoxlaması isinstance-dan daha sərtdir
    # n < 1 şərti həm mənfi ədədləri, həm də 0-ı əhatə edir
    if type(n) is not int or n < 1:
        return None

    # Düstur: n(n + 1)(2n + 1) / 6
    return (n * (n + 1) * (2 * n + 1)) // 6
