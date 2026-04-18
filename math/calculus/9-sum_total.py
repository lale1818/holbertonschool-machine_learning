#!/usr/bin/env python3
"""
Module for calculating the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n
    Args:
        n: ending condition
    Returns: integer sum or None
    """
    if not isinstance(n, (int, float)) or n < 0:
        return None
    
    # n-i tam ədədə çeviririk (n=5.0 kimi hallar üçün)
    n = int(n)
    
    # Kvadratlar cəmi düsturu: n(n + 1)(2n + 1) / 6
    return (n * (n + 1) * (2 * n + 1)) // 6
