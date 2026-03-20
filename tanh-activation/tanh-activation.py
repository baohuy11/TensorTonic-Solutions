import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x)
    ex1 = np.exp(x)
    ex2 = np.exp(-x)
    return (ex1 - ex2) / (ex1 + ex2)