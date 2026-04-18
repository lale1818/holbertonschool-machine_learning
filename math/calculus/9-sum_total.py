#!/usr/bin/env python3
"""
Bu modul n-ə qədər olan i kvadratlarının cəmini hesablayır.
"""


def summation_i_squared(n):
    """
    1-dən n-ə qədər olan i^2 cəmini dövr istifadə etmədən hesablayır.
    """
    if not isinstance(n, int) or n < 0:
        return None

    # Python-da vurma işarəsi (*) mütləqdir
    sum_total = (n * (n + 1) * (2 * n + 1)) // 6
    return sum_total
