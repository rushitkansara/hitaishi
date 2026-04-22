
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional

class PhysiologicalOscillator:
    """
    Layer 1: Intrinsic Physiological Oscillator.
    Handles fBM (Fractional Brownian Motion), RSA (Respiratory Sinus Arrhythmia),
    and Mayer Waves (0.1 Hz slow waves).
    """
    
    def __init__(self, n: int = 600, h: float = 0.75, l_matrix: Optional[np.ndarray] = None):
        self.n = n
        self.h = h
        self.l_matrix = l_matrix if l_matrix is not None else self._load_cholesky()
        
    def _load_cholesky(self) -> np.ndarray:
        """Loads precomputed Cholesky matrix for noise generation."""
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_path, 'fbm_cholesky_600.pkl')
        if not os.path.exists(path):
            from .precompute import precompute_cholesky
            return precompute_cholesky(self.n, self.h, path)
        with open(path, 'rb') as f:
            return pickle.load(f)

    def generate_fgn(self, sigma: float = 1.0) -> np.ndarray:
        """Generates Fractional Gaussian Noise using precomputed L matrix."""
        z = np.random.standard_normal(self.n)
        # Noise = L * Z
        noise = self.l_matrix @ z
        return noise * sigma

    def rsa_term(self, rr_array: np.ndarray, amplitude: float = 3.5) -> np.ndarray:
        """
        Generates Respiratory Sinus Arrhythmia (HR fluctuations synced to RR).
        HR fluctuates ~4-8 bpm with the breathing cycle.
        """
        # Integrate RR to get phase
        dt = 1.0 # 1Hz
        phase = np.cumsum(2 * np.pi * rr_array / 60.0 * dt)
        return amplitude * np.sin(phase)

    def mayer_wave(self, amplitude: float = 1.5) -> np.ndarray:
        """
        Generates 0.1 Hz Mayer waves (slow waves in blood pressure and heart rate).
        """
        t = np.arange(self.n)
        return amplitude * np.sin(2 * np.pi * 0.1 * t)

    def respiratory_cycle(self, rr_base: float, sigma: float = 0.4) -> np.ndarray:
        """
        Generates the basic respiratory drive with irregularity and sighs.
        """
        t = np.arange(self.n)
        # Base cycle
        cycle = 0.8 * np.sin(2 * np.pi * t / (60.0 / rr_base))
        # Add long-memory noise
        noise = self.generate_fgn(sigma=sigma)
        
        # Periodic "Sigh" every ~60-90s
        # (Simplified: add a Gaussian pulse)
        for i in range(1, self.n // 60):
            center = i * 60 + np.random.randint(-10, 10)
            if center < self.n:
                cycle += 2.5 * np.exp(-((t - center)**2) / (2 * 1.5**2))
                
        return cycle

if __name__ == '__main__':
    # Quick test
    osc = PhysiologicalOscillator(n=600, h=0.75)
    rr_base = 16.0
    rr_vals = rr_base + osc.respiratory_cycle(rr_base)
    hr_rsa = osc.rsa_term(rr_vals)
    hr_noise = osc.generate_fgn(sigma=1.2)
    mayer = osc.mayer_wave()
    
    print(f"Generated Oscillator test. HR Noise Std: {np.std(hr_noise):.4f}")
    print(f"RSA Max/Min: {np.max(hr_rsa):.2f} / {np.min(hr_rsa):.2f}")
