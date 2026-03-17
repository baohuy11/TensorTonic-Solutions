import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)
    
    # Step 1: Update First Moment (m_t)
    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    m_new = beta1 * m + (1 - beta1) * grad
    
    # Step 2: Update Second Moment (v_t)
    # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    v_new = beta2 * v + (1 - beta2) * (grad**2)
    
    # Step 3: Nesterov-Adjusted Update
    # The term (beta1 * m_new + (1 - beta1) * grad) provides the Nesterov "look-ahead"
    # combined with the adaptive learning rate from sqrt(v_new)
    nesterov_term = (beta1 * m_new + (1 - beta1) * grad)
    w_new = w - lr * nesterov_term / (np.sqrt(v_new) + eps)
    
    return w_new, m_new, v_new