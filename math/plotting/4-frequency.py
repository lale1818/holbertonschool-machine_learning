#!/usr/bin/env python3
"""
Plots a histogram of student scores for a project
"""
import matplotlib.pyplot as plt
import numpy as np


def frequency():
    """ Plots a histogram with 40-100 bins and explicit 40-100 x-limits """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = list(range(40, 101, 10))
    plt.hist(student_grades, bins=bins, edgecolor='black')
    
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.xlim(40, 100)
    plt.show()
