import numpy as np
import torch

def normalize_data(data):
    """Normalize data to have zero mean and unit variance."""
    return (data - np.mean(data)) / np.std(data)

def to_tensor(numpy_array):
    """Convert a NumPy array to a PyTorch tensor."""
    return torch.tensor(numpy_array, dtype=torch.float32)