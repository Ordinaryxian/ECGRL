import torch
from torch_geometric.data import Data
from itertools import permutations
from matplotlib import pyplot as plt
import numpy as np
import random
def seed_torch(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True