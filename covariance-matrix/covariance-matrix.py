import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.array(X)
    X = np.atleast_2d(X)
    
    N, D = X.shape
    if N <= 1:
        return None
        
    mu = np.mean(X, axis=0)
    X_cen = X - mu
    cov = (1 / (N - 1)) * (X_cen.T @ X_cen)
    return cov