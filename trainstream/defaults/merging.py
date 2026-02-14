import numpy as np

def merge_lists(buffer, batch):
    """Simple concatenation for Python lists."""
    return buffer + batch

def merge_numpy(buffer, batch):
    """Concatenation for NumPy arrays (axis 0)."""
    return np.concatenate((buffer, batch), axis=0)

# Optional: Only import torch if installed
try:
    import torch
    def merge_torch(buffer, batch):
        """Concatenation for PyTorch tensors (dim 0)."""
        return torch.cat((buffer, batch), dim=0)
except ImportError:
    pass