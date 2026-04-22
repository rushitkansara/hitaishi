
import numpy as np
import os
import random
import argparse
from typing import List, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count
from .engine import VitalsGeneratorV2
from .clinical_logic import ClinicalStateEngine

# Global variable to hold matrix in worker memory
GLOBAL_L_MATRIX = None

def init_worker(l_matrix):
    global GLOBAL_L_MATRIX
    GLOBAL_L_MATRIX = l_matrix

def generate_single_sample(args: Tuple[int, int]) -> Tuple[np.ndarray, int]:
    """Worker function for parallel generation."""
    state_code, seed = args
    random.seed(seed)
    np.random.seed(seed)
    
    # ... randomization logic ...
    age = random.uniform(18, 85)
    gender = random.choice(['male', 'female'])
    fitness = random.uniform(0.1, 0.9)
    
    potential_conditions = ['Hypertension', 'COPD', 'Diabetes', 'Obesity', 'Heart Failure']
    n_cond = random.randint(0, 2)
    conditions = random.sample(potential_conditions, n_cond)
    
    gen = VitalsGeneratorV2(age=age, gender=gender, initial_state=state_code, 
                           fitness=fitness, conditions=conditions, duration=600,
                           l_matrix=GLOBAL_L_MATRIX)
    data, _ = gen.generate_episode()
    return data, state_code

def create_v2_dataset(samples_per_class: int = 4700, test_split_ratio: float = 0.3, 
                      output_dir: str = 'data_v2/', max_cores: int = None):
    """Generates a balanced v2.0 dataset with frequent feedback and ETA."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    states = ClinicalStateEngine.STATES
    
    # 1. PRE-LOAD CHOLESKY MATRIX
    from .precompute import precompute_cholesky
    dir_path = os.path.dirname(os.path.abspath(__file__))
    l_path = os.path.join(dir_path, 'fbm_cholesky_600.pkl')
    l_matrix = precompute_cholesky(n=600, h=0.75, cache_path=l_path)
    
    tasks = []
    total_expected = len(states) * samples_per_class
    print(f"--- Hitaishi V2.0: Large-Scale Generation ---")
    print(f"Target: {total_expected} episodes | Windows: 600s @ 1Hz")
    
    for state_code in states.keys():
        for i in range(samples_per_class):
            tasks.append((state_code, random.randint(0, 10000000)))
            
    num_cores = max_cores if max_cores else cpu_count()
    print(f"Active Cores: {num_cores} | Progress Update: Every 50 samples")
    
    results = []
    import time
    import sys
    start_time = time.time()
    
    # Initialize pool with shared matrix
    with Pool(num_cores, initializer=init_worker, initargs=(l_matrix,)) as p:
        for i, res in enumerate(p.imap_unordered(generate_single_sample, tasks)):
            results.append(res)
            
            if (i + 1) % 50 == 0 or (i + 1) == total_expected:
                elapsed = time.time() - start_time
                eps = (i + 1) / elapsed
                remaining = (total_expected - (i + 1)) / eps
                
                # Simple progress bar string
                percent = ((i + 1) / total_expected) * 100
                bar_len = 20
                filled_len = int(bar_len * (i + 1) // total_expected)
                bar = '=' * filled_len + '-' * (bar_len - filled_len)
                
                # Format time
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                
                sys.stdout.write(f"\r[{bar}] {percent:4.1f}% | {i+1}/{total_expected} | {eps:4.1f} eps | ETA: {eta_min}m {eta_sec}s   ")
                sys.stdout.flush()
        
    print(f"\n\nGeneration Complete in {int((time.time()-start_time)//60)}m {int((time.time()-start_time)%60)}s")
    
    X = np.array([r[0] for r in results], dtype=np.float32)
    y = np.array([r[1] for r in results], dtype=np.int32)
    
    # Shuffle and Split...
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int((1 - test_split_ratio) * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    np.savez_compressed(os.path.join(output_dir, 'v2_train.npz'), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_dir, 'v2_test.npz'), X=X_test, y=y_test)
    print(f"Dataset Saved: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=4700)
    parser.add_argument('--cores', type=int, default=None, help='Limit CPU cores')
    args = parser.parse_args()
    
    create_v2_dataset(samples_per_class=args.samples, max_cores=args.cores)
