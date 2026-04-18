#!/usr/bin/env python3
"""
Module to calculate the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n without using loops
    Args:
        n: the stopping condition
    Returns:
        The integer value of the sum, or None if n is invalid
    """
    if not isinstance(n, int) or n < 0:
        return None

    # Kvadratlar cəmi düsturu: n(n + 1)(2n + 1) / 6
    # Bizdən integer cavab istədiyi üçün tam bölmə (//) istifadə edirik
    return (n * (n + 1) * (2 * n + 1)) // 6
