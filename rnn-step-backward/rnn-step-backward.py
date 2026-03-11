import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h  (shape: H,)
        dW: gradient wrt W                (shape: H x D)
        dU: gradient wrt U                (shape: H x H)
        db: gradient wrt bias             (shape: H,)
    """
    x_t, h_prev, h_t, W, U, b = [np.array(v) for v in cache]
    dh = np.array(dh)
    
    d_tanh = dh * (1 - h_t ** 2)
    
    dW = np.outer(d_tanh, x_t)
    dU = np.outer(d_tanh, h_prev)
    
    db = d_tanh
    
    dx_t = np.dot(W.T, d_tanh)
    dh_prev = np.dot(U.T, d_tanh)
    
    return dx_t, dh_prev, dW, dU, db
    
