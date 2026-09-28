import numpy as np


def get_xor_data():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ], dtype=float)

    return X, y