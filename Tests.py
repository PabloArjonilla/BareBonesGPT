import torch
import torch.nn as nn
from torch.nn import functional as F

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(mps_device)
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")