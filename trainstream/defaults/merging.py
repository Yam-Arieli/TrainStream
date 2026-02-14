__all__ = ["merge_lists", "merge_numpy"] 

import numpy as np
import torch

def merge_lists(buffer, batch):
    """Simple concatenation for Python lists."""
    return buffer + batch

def merge_numpy(buffer, batch):
    """Concatenation for NumPy arrays (axis 0)."""
    return np.concatenate((buffer, batch), axis=0)