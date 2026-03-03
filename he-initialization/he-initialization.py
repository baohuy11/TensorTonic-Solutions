import numpy as np
import array
def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    W = np.array(W)
    bound = np.sqrt(6.0 / fan_in)
    W_he = (W * 2 - 1) * bound
    return W_he