import torch
import numpy as np 
import random 


def init_fn(worker_id):
    np.random.seed(int(SEED))
    random.seed(int(SEED))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


global SEED 
SEED=32
set_seed(SEED)
