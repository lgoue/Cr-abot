import numpy as np


def softmax(x):
    try :
        return np.exp(x) / sum(np.exp(x) + 0.001)
    except:
        return np.exp(x) / 0.001


def entropy(x):
    return x * np.log(x + 0.00001)
