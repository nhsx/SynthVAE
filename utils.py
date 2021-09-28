import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
