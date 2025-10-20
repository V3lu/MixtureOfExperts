# 1. Gating mechanism
# 2. Expert networks 
# 3. Output layer

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchvision import transforms, models

# ---------------------- CONFIG ----------------------
BATCH_SIZE = 64
EPOCHS = 6
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ HELPER FUNCTIONS ---------------------
def KeepTopK(v, k, i):
    if v[i] in k[k:]:
        return v[i]
    else:
        return 0


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.gating(x)
        return logits



class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




class Output(nn.Module):
    def __init__(self):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits






















