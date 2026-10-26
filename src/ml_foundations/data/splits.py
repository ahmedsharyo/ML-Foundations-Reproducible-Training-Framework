import numpy as np

def deterministic_split(size: int, validation_fraction: float=.2, seed: int=42):
    if size<2 or not 0<validation_fraction<1: raise ValueError("size >= 2 and fraction in (0,1) required")
    order=np.random.default_rng(seed).permutation(size); n=max(1,min(size-1,round(size*validation_fraction)))
    return order[n:].tolist(),order[:n].tolist()
