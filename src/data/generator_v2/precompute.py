
import numpy as np
import pickle
import os
from typing import Optional

def fbm_covariance(n: int, h: float) -> np.ndarray:
    """
    Computes the covariance matrix for Fractional Brownian Motion.
    
    Parameters:
        n: Number of time steps.
        h: Hurst exponent (0.5 = Brownian, >0.5 = Persistent).
        
    Returns:
        Covariance matrix of shape (n, n).
    """
    # Create an array of indices [0, 1, ..., n-1]
    t = np.arange(n)
    # Use broadcasting to compute all |t_i - t_j|
    diff = np.abs(t[:, None] - t[None, :])
    # FBM covariance formula: 0.5 * (|t|^2H + |s|^2H - |t-s|^2H)
    # For our incremental noise (Fractional Gaussian Noise), the covariance is:
    # γ(k) = 0.5 * (|k+1|^2H + |k-1|^2H - 2|k|^2H)
    
    # We actually want the covariance of the increments (Fractional Gaussian Noise)
    # so we can sum them later. This makes the noise stationary.
    k = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    gamma = 0.5 * (np.abs(k+1)**(2*h) + np.abs(k-1)**(2*h) - 2*np.abs(k)**(2*h))
    return gamma

def precompute_cholesky(n: int = 600, h: float = 0.75, cache_path: str = 'fbm_cholesky.pkl') -> np.ndarray:
    """
    Precomputes and caches the Cholesky decomposition of the FBM covariance matrix.
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print(f"Precomputing Cholesky for n={n}, H={h}...")
    cov = fbm_covariance(n, h)
    # Cholesky decomposition: cov = L * L^T
    # This allows generating noise: X = L * Z where Z ~ N(0, I)
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n)) # Add small epsilon for stability
    
    with open(cache_path, 'wb') as f:
        pickle.dump(L, f)
    
    return L

if __name__ == '__main__':
    # Default to 600s window and H=0.75
    target_dir = os.path.dirname(os.path.abspath(__file__))
    precompute_cholesky(n=600, h=0.75, cache_path=os.path.join(target_dir, 'fbm_cholesky_600.pkl'))
    print("Precomputation COMPLETE.")
