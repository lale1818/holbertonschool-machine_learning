#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for j in range(n):
        sub = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix"""
    n = len(matrix)
    if n == 1:
        return [[1]]
    adj = []
    for j in range(n):
        adj_row = []
        for i in range(n):
            sub = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            adj_row.append(((-1) ** (i + j)) * determinant(sub))
        adj.append(adj_row)
    return adj


def inverse(matrix):
    """Calculates the inverse of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    if n == 0 or (n == 1 and len(matrix[0]) == 0):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None
    
    if n == 1:
        return [[1 / matrix[0][0]]]

    adj = adjugate(matrix)
    return [[val / det for val in row] for row in adj]
