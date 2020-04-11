import random

import numpy as np


def generate_data(nums=30):
    x1 = np.array([[random.uniform(0, 5)] for _ in range(nums)])
    y1 = np.array([0 for _ in range(nums)])

    x2 = np.array([[random.uniform(3, 8)] for _ in range(nums)])
    y2 = np.array([1 for _ in range(nums)])
    return x1, y1, x2, y2
