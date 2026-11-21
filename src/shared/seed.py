"""One function to fix every source of randomness.

Call ``set_seed(seed)`` at the very start of every script. The team uses the
three seeds 13, 42 and 2027 (see configs/shared.yaml) for every table row.
"""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional: the scorer-free scripts run without it
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN is slower but makes runs repeatable.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass
