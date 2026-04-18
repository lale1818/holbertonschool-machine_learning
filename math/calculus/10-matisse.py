#!/usr/bin/env python3
"""
Çoxhədlinin törəməsini hesablayan modul
"""


def poly_derivative(poly):
    """
    Siyahı şəklində verilmiş çoxhədlinin törəməsini hesablayır.
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    for coeff in poly:
        if not isinstance(coeff, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    derivative = [poly[i] * i for i in range(1, len(poly))]

    while len(derivative) > 1 and derivative[-1] == 0:
        derivative.pop()

    return derivative
