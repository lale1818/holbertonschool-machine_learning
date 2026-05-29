#!/usr/bin/env python3
"""
Plots a stacked bar graph representing fruit quantities per person.
"""
import matplotlib.pyplot as plt
import numpy as np


def bars():
    """ Plots stacked bars for apples, bananas, oranges, and peaches """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    persons = ['Farrah', 'Fred', 'Felicia']
    width = 0.5

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(persons, apples, width=width, color='red', label='apples')
    plt.bar(persons, bananas, width=width, color='yellow', bottom=apples, label='bananas')
    plt.bar(persons, oranges, width=width, color='#ff8000', bottom=apples + bananas, label='oranges')
    plt.bar(persons, peaches, width=width, color='#ffe5b4', bottom=apples + bananas + oranges, label='peaches')

    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend()
    plt.show()
