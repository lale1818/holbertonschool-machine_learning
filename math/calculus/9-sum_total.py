#!/usr/bin/env python3
"""
Bu modul n-ə qədər olan i kvadratlarının cəmini hesablayır.
"""


def summation_i_squared(n):
    """
    1-dən n-ə qədər olan i^2 cəmini dövr istifadə etmədən hesablayır.
    Args:
        n (int): Dayandırılma şərti.
    Returns:
        int: Cəmin nəticəsi, əgər n düzgün rəqəm deyilsə None.
    """
    if not isinstance(n, (int, float)) or n < 0:
        return None
    
    # n float olarsa integer-ə çeviririk
    n = int(n)

    # Kvadratlar cəmi düsturu: n(n + 1)(2n + 1) / 6
    sum_total = (n * (n + 1) * (2 * n + 1)) // 6
    return sum_total
