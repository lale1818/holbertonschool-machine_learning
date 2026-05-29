#!/usr/bin/env python3
"""
Plots a histogram of student scores for a project
"""
import matplotlib.pyplot as plt
import numpy as np


def frequency():
    """ Plots a histogram with bins every 10 units from 40 to 100 """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    bins = np.arange(40, 101, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()
