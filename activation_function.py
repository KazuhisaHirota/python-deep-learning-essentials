import numpy as np


def step(x: float) -> int:
    if x >= 0.0:
        return 1
    else:
        return -1

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x: list, n: int) -> list:
    e = np.exp(x - np.max(x)) # to avoid overflow
    return e / np.sum(e)

