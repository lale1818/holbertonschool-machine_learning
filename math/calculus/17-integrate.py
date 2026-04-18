#!/usr/bin/env python3
"""
Module to calculate the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Calculates the integral of a polynomial
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    integral = [C]
    for i in range(len(poly)):
        if not isinstance(poly[i], (int, float)):
            return None
        value = poly[i] / (i + 1)
        if value % 1 == 0:
            value = int(value)
        integral.append(value)

    # Sondakı sıfırları təmizləyirik ki, [0] və ya [7] kimi nəticə qalsın
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
