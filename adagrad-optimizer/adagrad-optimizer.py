import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    new_w = []
    new_g = []
    for i in range(len(w)):
        updated_g = G[i] + (g[i] ** 2)
        new_g.append(updated_g)

        denom = np.sqrt(updated_g + eps)
        updated_w = w[i] - (lr / denom) * g[i]
        new_w.append(updated_w)

    return new_w, new_g