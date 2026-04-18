#!/usr/bin/env python3
"""
Module to calculate the sum of i squared
"""


def summation_i_squared(n):
    """
    Calculates the sum of i^2 from 1 to n without using loops
    """
    # n-in tam ədəd olub-olmadığını və 0-dan böyük (və ya bərabər) olduğunu yoxla
    # Əgər n < 1 olanda None gözləyirsə, şərti (n < 1) elə.
    # Amma çox vaxt (n < 0) və ya tip yoxlaması problem yaradır.
    if type(n) is not int or n < 0:
        return None

    # Riyazi düstur
    return (n * (n + 1) * (2 * n + 1)) // 6
