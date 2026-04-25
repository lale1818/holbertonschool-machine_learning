#!/usr/bin/env python3
"""
Module to perform matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat2[0])):
            dot_product = 0
            for k in range(len(mat2)):
                dot_product += mat1[i][k] * mat2[k][j]
            new_row.append(dot_product)
        result.append(new_row)

    return result
