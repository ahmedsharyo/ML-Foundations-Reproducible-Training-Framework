import os, random
import numpy as np
import torch

def set_seed(seed: int, deterministic: bool=True) -> None:
    os.environ["PYTHONHASHSEED"]=str(seed); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True,warn_only=True)
        if hasattr(torch.backends,"cudnn"): torch.backends.cudnn.benchmark=False

def get_device(requested: str="auto") -> torch.device:
    if requested=="auto": return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested=="mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS requested but unavailable")
    if requested not in ("cpu","mps"): raise ValueError("device must be auto, cpu or mps")
    return torch.device(requested)
