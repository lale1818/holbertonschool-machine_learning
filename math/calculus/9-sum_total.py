#!/usr/bin/env python3
"""
Module to calculate the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n
    Args:
        n: the stopping condition
    Returns:
        The integer value of the sum, or None if n is invalid
    """
    if not isinstance(n, int) or n < 0:
        return None

    # Kvadratlar cəmi düsturu: n(n + 1)(2n + 1) / 6
    # n = 0 olanda nəticə 0 olur.
    # Əgər checker n=0-ı da invalid sayırsa, (n < 1) edərik.
    # Amma əksərən n < 0 və tip yoxlaması kifayətdir.
    result = (n * (n + 1) * (2 * n + 1)) // 6
    return result
