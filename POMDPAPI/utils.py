import numpy as np


def softmax(x):
    return np.exp(x) / sum(np.exp(x) + 0.00001)


def entropy(x):
    return x * np.log(x + 0.00001)
