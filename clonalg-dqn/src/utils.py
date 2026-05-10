import random
import numpy as np
import torch

def set_seed(seed: int):
    """Fija todas las semillas para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def moving_average(values, window=10):
    """Media móvil para suavizar curvas de aprendizaje."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')