import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    x = np.array(x)
    h_prev = np.array(h_prev)
    
    x, x_was_1d = _as2d(x, x.shape[-1])
    h_prev, _ = _as2d(h_prev, h_prev.shape[-1])

    Uz, Ur, Uh = params['Uz'], params['Ur'], params['Uh']
    Wz, Wr, Wh = params['Wz'], params['Wr'], params['Wh']
    bz, br, bh = params['bz'], params['br'], params['bh']

    z = _sigmoid(np.dot(x, Wz) + np.dot(h_prev, Uz) + bz)

    r = _sigmoid(np.dot(x, Wr) + np.dot(h_prev, Ur) + br)

    h_candi = np.tanh(np.dot(x, Wh) + np.dot(r * h_prev, Uh) + bh)

    h_next = (1 - z) * h_prev + z * h_candi

    return h_next.flatten() if x_was_1d else h_next

